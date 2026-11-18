#!/usr/bin/env python3
"""
Generate RAMDocs-style conflict dataset.

This script generates a dataset with conflicting evidence for testing RAG systems.
All documents include a date, which is key for identifying temporal conflicts.

Document Types:
- correct: Documents with accurate answers (recent dates, high reliability)
- misinfo: Misinformation Conflict - incorrect/false information with SUBTLE
           uncertainty markers (hedging language, informal sources). The uncertainty
           is detectable but not glaringly obvious.
- temporal: Temporal Conflict - knowledge that was correct at the time but has since
            changed. Identified primarily by OLDER DATES compared to correct documents.
            No obvious "archived" or "outdated" labels - the date is the signal.
- noise: Noise - contains correct information that is irrelevant to the question
         (distractors from unrelated topics)

Date Strategy:
- correct: Recent (7-90 days ago)
- misinfo: Varied (14-180 days ago)  
- temporal: Old (6 months to 3 years ago)
- noise: Varied (7 days to 1 year ago)

Output format matches RAMDocs:
{
    "question": "Research question",
    "documents": [
        {"text": "...", "type": "correct", "answer": "...", "format": "md", 
         "document_date": "2024-11-15", ...},
        {"text": "...", "type": "misinfo", "answer": "...", "format": "email",
         "document_date": "2024-08-20", ...},
        ...
    ],
    "disambig_entity": ["Company/Context"],
    "gold_answers": ["correct answer"],
    "wrong_answers": ["incorrect answer 1", ...]
}
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from llm import prompt_llm, DEFAULT_MODEL
from generators.email_generator import generate_email_document
from generators.chat_generator import generate_mattermost_chat
from generators.md_generator import generate_markdown_document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_random_document_counts_multi(
    num_answers: int,
    min_correct_per_answer: int = 1,
    max_correct_per_answer: int = 3,
    min_misinfo_per_answer: int = 1,
    max_misinfo_per_answer: int = 2,
    min_temporal_per_answer: int = 0,
    max_temporal_per_answer: int = 1,
    min_noise: int = 2,
    max_noise: int = 4,
) -> Dict:
    """
    Generate per-answer document counts for multi-answer tasks.

    Returns:
        Dict with "per_answer" list (one entry per answer with correct/misinfo/temporal
        counts) and shared "num_noise".
    """
    per_answer = []
    for _ in range(num_answers):
        per_answer.append({
            "num_correct": random.randint(min_correct_per_answer, max_correct_per_answer),
            "num_misinfo": random.randint(min_misinfo_per_answer, max_misinfo_per_answer),
            "num_temporal": random.randint(min_temporal_per_answer, max_temporal_per_answer),
        })

    return {
        "per_answer": per_answer,
        "num_noise": random.randint(min_noise, max_noise),
    }


def generate_random_document_counts(
    min_total: int = 5,
    max_total: int = 10,
    min_correct: int = 1,
    max_correct: int = 1,
) -> Dict[str, int]:
    """
    Generate random document counts.
    
    Design:
    - Each task has exactly 1 ground-truth fact (correct document)
    - Misinfo and temporal conflicts are generated from the same ground-truth fact
    - Noise comes from distractors (contains correct but irrelevant information)
    
    Returns:
        Dict with num_correct, num_misinfo, num_temporal, num_noise
    """
    # Total docs between min_total and max_total
    total_docs = random.randint(min_total, max_total)
    
    # Correct docs (default 1 per task since each task is now 1 fact)
    num_correct = random.randint(min_correct, max_correct)
    remaining = total_docs - num_correct
    
    # At least 1 noise (from distractors - correct but irrelevant info)
    num_noise = random.randint(1, max(1, min(remaining // 2, 4)))
    remaining -= num_noise
    
    # Distribute rest among misinfo and temporal
    if remaining > 0:
        conflict_types = ["misinfo", "temporal"]
        random.shuffle(conflict_types)
        
        num_misinfo = 0
        num_temporal = 0
        
        for ctype in conflict_types:
            if remaining <= 0:
                break
            count = random.randint(0, min(remaining, 3))
            if ctype == "misinfo":
                num_misinfo = count
            else:
                num_temporal = count
            remaining -= count
    else:
        num_misinfo = 0
        num_temporal = 0
    
    return {
        "num_correct": num_correct,
        "num_misinfo": num_misinfo,
        "num_temporal": num_temporal,
        "num_noise": num_noise,
    }


class DocumentType(str, Enum):
    CORRECT = "correct"      # Accurate, verified information
    MISINFO = "misinfo"      # Misinformation with uncertainty/unreliability markers
    TEMPORAL = "temporal"    # Temporal conflict - outdated/superseded information
    NOISE = "noise"          # Correct but irrelevant information


class DocumentFormat(str, Enum):
    """Document formats that can be represented as text in JSONL output."""
    EMAIL = "email"   # JSONL format
    CHAT = "chat"     # JSONL format (Mattermost)
    MD = "md"         # Markdown text


# Document type to format probability mapping
# Misinfo tends toward informal sources (email, chat) to reflect unreliability
# Temporal can appear in any format (old docs, archived emails, etc.)
# Noise can be any format (correct info from unrelated topics)
FORMAT_PROBABILITIES = {
    DocumentType.CORRECT: {
        DocumentFormat.MD: 0.50,      # Official documentation
        DocumentFormat.EMAIL: 0.30,   # Formal communications
        DocumentFormat.CHAT: 0.20,    # Verified discussions
    },
    DocumentType.MISINFO: {
        DocumentFormat.EMAIL: 0.40,   # Informal emails with uncertainty
        DocumentFormat.CHAT: 0.45,    # Rumors, unverified chats
        DocumentFormat.MD: 0.15,      # Draft/unverified docs
    },
    DocumentType.TEMPORAL: {
        DocumentFormat.MD: 0.40,      # Archived documentation
        DocumentFormat.EMAIL: 0.35,   # Old email threads
        DocumentFormat.CHAT: 0.25,    # Historical discussions
    },
    DocumentType.NOISE: {
        DocumentFormat.MD: 0.40,      # Unrelated docs
        DocumentFormat.EMAIL: 0.30,   # Off-topic emails
        DocumentFormat.CHAT: 0.30,    # Tangential discussions
    },
}


@dataclass
class Fact:
    """Represents a fact from the CSV file."""
    fact_id: str
    fact_type: str  # supporting or distractor
    source: str  # internal or external
    question: str
    answer: str
    dr_question: str
    
    def to_dict(self) -> Dict:
        return {
            "fact_id": self.fact_id,
            "fact_type": self.fact_type,
            "source": self.source,
            "question": self.question,
            "answer": self.answer,
        }


@dataclass
class ModifiedFact:
    """Represents a modified version of a fact (for misinfo or temporal conflicts)."""
    original_fact: Fact
    modified_answer: str
    modification_type: str  # "misinfo" or "temporal"
    modification_reason: str
    
    def to_dict(self) -> Dict:
        return {
            "original_fact_id": self.original_fact.fact_id,
            "original_answer": self.original_fact.answer,
            "modified_answer": self.modified_answer,
            "modification_type": self.modification_type,
            "modification_reason": self.modification_reason,
        }


@dataclass
class Document:
    """Represents a generated document."""
    text: str
    doc_type: str
    answer: str
    doc_format: str
    document_date: str  # ISO format date string (YYYY-MM-DD)
    source_fact: Optional[Dict] = None
    reliability_signals: Optional[Dict] = None
    file_path: Optional[str] = None
    confidence: Optional[float] = None  # Ground-truth confidence score [0, 1]


# Ground-truth confidence ranges by document type.
# Confidence reflects how confidently the *document writer* asserts the claim,
# NOT whether the claim helps answer the target question.
# - correct: confident AND accurate
# - misinfo: low confidence (writer hedges with uncertainty markers)
# - temporal: high confidence (writer was confident at the time; date is the
#   signal that it's outdated, not confidence)
# - noise: high confidence (writer asserts a true fact, just on a different
#   topic; off-topic is the signal, not confidence)
CONFIDENCE_RANGES = {
    DocumentType.CORRECT: (0.80, 1.00),
    DocumentType.MISINFO: (0.10, 0.35),
    DocumentType.TEMPORAL: (0.70, 0.95),
    DocumentType.NOISE: (0.75, 0.95),
}


def sample_confidence(doc_type: DocumentType) -> float:
    """Sample a ground-truth confidence score for a document type."""
    lo, hi = CONFIDENCE_RANGES[doc_type]
    return round(random.uniform(lo, hi), 2)


@dataclass
class ConflictTask:
    """Represents a complete conflict task in RAMDocs format."""
    question: str
    documents: List[Dict]
    disambig_entity: List[str]
    gold_answers: List[str]
    wrong_answers: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "documents": self.documents,
            "disambig_entity": self.disambig_entity,
            "gold_answers": self.gold_answers,
            "wrong_answers": self.wrong_answers,
        }


def select_format_by_probability(doc_type: DocumentType) -> DocumentFormat:
    """Select a document format based on probability distribution for the doc type."""
    probs = FORMAT_PROBABILITIES[doc_type]
    formats = list(probs.keys())
    weights = list(probs.values())
    return random.choices(formats, weights=weights, k=1)[0]


def generate_document_date(doc_type: DocumentType, reference_date: Optional[datetime] = None) -> str:
    """
    Generate an appropriate date for a document based on its type.
    
    Date strategy:
    - correct: Recent (within last 1-3 months from reference_date)
    - misinfo: Varied (can be recent or old, adds to confusion)
    - temporal: Old (6 months to 3 years before reference_date)
    - noise: Varied (realistic spread, independent of conflict)
    
    Args:
        doc_type: The type of document
        reference_date: The reference "current" date (defaults to today)
    
    Returns:
        ISO format date string (YYYY-MM-DD)
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    if doc_type == DocumentType.CORRECT:
        # Recent: 7-90 days ago
        days_ago = random.randint(7, 90)
        doc_date = reference_date - timedelta(days=days_ago)
    
    elif doc_type == DocumentType.MISINFO:
        # Varied: can be recent or moderately old (adds uncertainty)
        # 14-180 days ago
        days_ago = random.randint(14, 180)
        doc_date = reference_date - timedelta(days=days_ago)
    
    elif doc_type == DocumentType.TEMPORAL:
        # Old: 6 months to 3 years ago (clearly outdated)
        days_ago = random.randint(180, 1095)
        doc_date = reference_date - timedelta(days=days_ago)
    
    elif doc_type == DocumentType.NOISE:
        # Varied: realistic spread (7 days to 1 year ago)
        days_ago = random.randint(7, 365)
        doc_date = reference_date - timedelta(days=days_ago)
    
    else:
        # Default to recent
        days_ago = random.randint(7, 60)
        doc_date = reference_date - timedelta(days=days_ago)
    
    return doc_date.strftime("%Y-%m-%d")


def format_date_for_document(date_str: str, doc_format: DocumentFormat) -> str:
    """
    Format a date string appropriately for embedding in document content.
    
    Different formats use different date conventions:
    - Email: "Mon, DD Mon YYYY" or similar email header format
    - Chat: Casual format or timestamp
    - MD: Various formal formats
    
    Args:
        date_str: ISO format date (YYYY-MM-DD)
        doc_format: The document format
    
    Returns:
        Formatted date string suitable for the document type
    """
    doc_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    if doc_format == DocumentFormat.EMAIL:
        # Email header style: "Mon, 15 Jan 2024" or "January 15, 2024"
        formats = [
            doc_date.strftime("%a, %d %b %Y"),
            doc_date.strftime("%B %d, %Y"),
            doc_date.strftime("%m/%d/%Y"),
        ]
        return random.choice(formats)
    
    elif doc_format == DocumentFormat.CHAT:
        # Chat timestamp style: casual
        formats = [
            doc_date.strftime("%m/%d/%y"),
            doc_date.strftime("%b %d"),
            doc_date.strftime("%Y-%m-%d"),
        ]
        return random.choice(formats)
    
    else:  # MD
        # Formal document style
        formats = [
            doc_date.strftime("%B %d, %Y"),
            doc_date.strftime("%Y-%m-%d"),
            doc_date.strftime("%d %B %Y"),
        ]
        return random.choice(formats)


class ConflictDatasetGenerator:
    """Generate RAMDocs-style conflict datasets from facts."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        logger.info(f"Using model: {model}")
        
    def load_facts_from_csv(self, csv_path: str) -> Tuple[str, List[Fact], List[Fact]]:
        """
        Load facts from a CSV file.
        
        Returns:
            Tuple of (dr_question, supporting_facts, distractor_facts)
        """
        supporting_facts = []
        distractor_facts = []
        dr_question = None
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fact = Fact(
                    fact_id=row['fact_id'],
                    fact_type=row['type'],
                    source=row['external_or_internal'],
                    question=row['question'],
                    answer=row['answer'],
                    dr_question=row['dr_question'],
                )
                dr_question = row['dr_question']
                
                if row['type'] == 'supporting':
                    supporting_facts.append(fact)
                else:
                    distractor_facts.append(fact)
        
        return dr_question, supporting_facts, distractor_facts
    
    def _generate_with_file_generator(
        self,
        doc_format: DocumentFormat,
        fact_dict: Dict,
        company_info: Dict,
        persona: Dict,
        output_dir: str,
    ) -> Tuple[str, str]:
        """
        Generate document using the file generators.
        
        Returns:
            Tuple of (document_text, file_path)
        """
        # Create temp output directory
        temp_dir = tempfile.mkdtemp(dir=output_dir)
        
        # Build topic string from fact dict
        topic = f"""Topic: {fact_dict.get('category', 'internal')}
Insight: {fact_dict['insight']}
Question: {fact_dict['validation_question']}
Answer: {fact_dict['answer']}
Context: {fact_dict.get('additional_context', '')}"""
        
        # Generate document based on format
        if doc_format == DocumentFormat.EMAIL:
            file_path = generate_email_document(
                topic=topic,
                output_dir=temp_dir,
                persona=persona,
                company_info=company_info,
                word_limit=300,
                model=self.model,
            )
        elif doc_format == DocumentFormat.CHAT:
            file_path = generate_mattermost_chat(
                topic=topic,
                output_dir=temp_dir,
                company_info=company_info,
                word_limit=300,
                model=self.model,
            )
        elif doc_format == DocumentFormat.MD:
            file_path = generate_markdown_document(
                topic=topic,
                output_path=temp_dir,
                company_info=company_info,
                word_limit=500,
                model=self.model,
            )
        else:
            raise ValueError(f"Unsupported document format: {doc_format}")
        
        # Read the generated file content
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return text, file_path
    
    def _create_fact_dict_for_generator(
        self,
        fact: Fact,
        modified_answer: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> Dict:
        """Create a fact dictionary compatible with generators."""
        answer = modified_answer if modified_answer else fact.answer
        
        return {
            "id": fact.fact_id,
            "category": "internal" if fact.source == "internal" else "external",
            "insight": f"{fact.question} {answer}",
            "validation_question": fact.question,
            "answer": answer,
            "additional_context": additional_context or "",
        }
    
    def generate_correct_document(
        self,
        fact: Fact,
        company_info: Dict,
        persona: Dict,
        output_dir: str,
        reference_date: Optional[datetime] = None,
    ) -> Document:
        """Generate a document with the correct answer and high reliability signals."""
        doc_format = select_format_by_probability(DocumentType.CORRECT)
        logger.info(f"  Format: {doc_format.value}")
        
        # Generate a recent date for correct documents
        doc_date = generate_document_date(DocumentType.CORRECT, reference_date)
        formatted_date = format_date_for_document(doc_date, doc_format)
        
        fact_dict = self._create_fact_dict_for_generator(
            fact,
            additional_context=f"Document date: {formatted_date}. This is verified and official information."
        )
        
        text, file_path = self._generate_with_file_generator(
            doc_format, fact_dict, company_info, persona, output_dir
        )
        
        return Document(
            text=text,
            doc_type=DocumentType.CORRECT.value,
            answer=fact.answer,
            doc_format=doc_format.value,
            document_date=doc_date,
            source_fact=fact.to_dict(),
            reliability_signals={
                "recency": "current",
                "verification": "verified",
                "authority": "high",
            },
            file_path=file_path,
            confidence=sample_confidence(DocumentType.CORRECT),
        )

    def generate_misinfo_document(
        self,
        fact: Fact,
        company_info: Dict,
        persona: Dict,
        output_dir: str,
        reference_date: Optional[datetime] = None,
    ) -> Tuple[Document, str]:
        """
        Generate a document with plausible but WRONG answer.
        
        Misinformation Conflict: Incorrect/false information with subtle uncertainty
        markers or unreliable source indicators embedded in the text.
        
        Uncertainty markers include:
        - Hedging language: "I heard", "I think", "supposedly", "might be"
        - Informal attributions: "someone mentioned", casual tone
        - Confidence caveats: "not 100% sure", "take this with a grain of salt"
        """
        doc_format = select_format_by_probability(DocumentType.MISINFO)
        logger.info(f"  Format: {doc_format.value}")
        
        # Generate date for this document
        doc_date = generate_document_date(DocumentType.MISINFO, reference_date)
        formatted_date = format_date_for_document(doc_date, doc_format)
        
        wrong_answer, modification_reason = self._generate_wrong_answer(fact, doc_date=doc_date)
        uncertainty_context = self._get_uncertainty_context(doc_format, formatted_date)
        
        fact_dict = self._create_fact_dict_for_generator(
            fact,
            modified_answer=wrong_answer,
            additional_context=uncertainty_context
        )
        
        text, file_path = self._generate_with_file_generator(
            doc_format, fact_dict, company_info, persona, output_dir
        )
        
        modified_fact = ModifiedFact(
            original_fact=fact,
            modified_answer=wrong_answer,
            modification_type="misinfo",
            modification_reason=modification_reason,
        )
        
        # Determine uncertainty type for metadata
        uncertainty_type = self._get_uncertainty_type(doc_format)
        
        return Document(
            text=text,
            doc_type=DocumentType.MISINFO.value,
            answer=wrong_answer,
            doc_format=doc_format.value,
            document_date=doc_date,
            source_fact=modified_fact.to_dict(),
            reliability_signals={
                "recency": "unclear",
                "verification": "unverified",
                "authority": "low",
                "uncertainty_type": uncertainty_type,
                "source_type": "informal" if doc_format in [DocumentFormat.EMAIL, DocumentFormat.CHAT] else "draft",
            },
            file_path=file_path,
            confidence=sample_confidence(DocumentType.MISINFO),
        ), wrong_answer
    
    def _get_uncertainty_context(self, doc_format: DocumentFormat, formatted_date: str) -> str:
        """
        Get appropriate uncertainty context based on document format.
        
        Returns instructions for the generator to embed SUBTLE uncertainty markers.
        The uncertainty should be detectable but not glaringly obvious.
        """
        uncertainty_phrases = {
            DocumentFormat.EMAIL: [
                "Use casual hedging like 'I think' or 'from what I understand'.",
                "Mention hearing this from a colleague or in passing.",
                "Use phrases like 'if I remember correctly' or 'I believe'.",
                "Frame it as informal knowledge sharing, not official communication.",
                "Include a casual tone suggesting this is secondhand information.",
            ],
            DocumentFormat.CHAT: [
                "Use casual language like 'pretty sure' or 'I think so'.",
                "Frame it as something heard from someone else.",
                "Include natural conversational uncertainty like 'might be' or 'could be'.",
                "Use informal phrasing that suggests incomplete knowledge.",
                "Add natural hedging like 'afaik' or 'last I heard'.",
            ],
            DocumentFormat.MD: [
                "Include subtle qualifying language like 'preliminary findings' or 'initial assessment'.",
                "Use phrases like 'based on available information' or 'current understanding'.",
                "Frame conclusions tentatively with 'appears to' or 'suggests that'.",
                "Include a brief note about information being subject to verification.",
                "Use conditional language like 'may' or 'could' for key claims.",
            ],
        }
        
        phrases = uncertainty_phrases.get(doc_format, uncertainty_phrases[DocumentFormat.EMAIL])
        selected = random.sample(phrases, min(2, len(phrases)))
        
        base_context = f"Document date: {formatted_date}. "
        base_context += "This contains INCORRECT information with subtle uncertainty markers. "
        base_context += " ".join(selected)
        base_context += " Keep uncertainty natural and subtle, not obvious."
        
        return base_context
    
    def _get_uncertainty_type(self, doc_format: DocumentFormat) -> str:
        """Get the type of uncertainty marker for metadata."""
        uncertainty_types = {
            DocumentFormat.EMAIL: random.choice(["hedging_language", "informal_source", "unverified_claim"]),
            DocumentFormat.CHAT: random.choice(["rumor", "secondhand_info", "low_confidence"]),
            DocumentFormat.MD: random.choice(["draft_document", "pending_review", "unconfirmed_data"]),
        }
        return uncertainty_types.get(doc_format, "unverified")
    
    def _generate_wrong_answer(
        self, fact: Fact, doc_date: Optional[str] = None
    ) -> Tuple[str, str]:
        """Generate a plausible but wrong answer."""
        date_constraint = (
            f"\n- The document containing this answer is dated {doc_date}. "
            "Any dates or time periods in the wrong answer MUST be after this document date "
            "(i.e. cannot reference events that have already passed relative to the document date)."
            if doc_date else ""
        )
        prompt = f"""Given this correct answer, generate a WRONG but plausible alternative.

CORRECT ANSWER: {fact.answer}
QUESTION: {fact.question}

RULES:
- If the answer contains a number/percentage, change it significantly (e.g., 8% -> 15% or 3%)
- If it contains a date, shift it by 6-18 months in either direction{date_constraint}
- Keep the same format/type as the original
- Make it plausible but clearly different

Return JSON:
{{"wrong_answer": "The wrong answer", "reason": "Brief explanation of what was changed"}}"""

        response = prompt_llm(prompt, model=self.model)
        try:
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
            data = json.loads(response)
            return data.get("wrong_answer", fact.answer), data.get("reason", "Modified value")
        except:
            return f"[Modified] {fact.answer}", "Fallback modification"
    
    def generate_temporal_document(
        self,
        fact: Fact,
        company_info: Dict,
        persona: Dict,
        output_dir: str,
        reference_date: Optional[datetime] = None,
    ) -> Tuple[Document, str]:
        """
        Generate a document with TEMPORAL CONFLICT - outdated/superseded information.
        
        Temporal Conflict: Knowledge that was once correct but has since changed or
        evolved over time. The key signal is the document DATE - temporal documents
        have older dates than correct documents.
        
        The content should read naturally for its time period without obvious
        "ARCHIVED" or "OUTDATED" labels. The reader/LLM must infer staleness
        from the date and context.
        """
        doc_format = select_format_by_probability(DocumentType.TEMPORAL)
        logger.info(f"  Format: {doc_format.value}")
        
        # Generate an OLD date for temporal documents (6 months to 3 years ago)
        doc_date = generate_document_date(DocumentType.TEMPORAL, reference_date)
        formatted_date = format_date_for_document(doc_date, doc_format)

        # Pass the document date so the temporal answer references a period
        # consistent with documents written on that date.
        outdated_answer, modification_reason = self._generate_temporal_answer(
            fact, doc_date=doc_date
        )
        temporal_context = self._get_temporal_context(doc_format, formatted_date)
        
        fact_dict = self._create_fact_dict_for_generator(
            fact,
            modified_answer=outdated_answer,
            additional_context=temporal_context
        )
        
        text, file_path = self._generate_with_file_generator(
            doc_format, fact_dict, company_info, persona, output_dir
        )
        
        modified_fact = ModifiedFact(
            original_fact=fact,
            modified_answer=outdated_answer,
            modification_type="temporal",
            modification_reason=modification_reason,
        )
        
        return Document(
            text=text,
            doc_type=DocumentType.TEMPORAL.value,
            answer=outdated_answer,
            doc_format=doc_format.value,
            document_date=doc_date,
            source_fact=modified_fact.to_dict(),
            reliability_signals={
                "recency": "outdated",
                "verification": "was_verified",
                "authority": "medium",
            },
            file_path=file_path,
            confidence=sample_confidence(DocumentType.TEMPORAL),
        ), outdated_answer
    
    def _get_temporal_context(self, doc_format: DocumentFormat, formatted_date: str) -> str:
        """
        Get appropriate temporal context based on document format.
        
        The temporal signal should be SUBTLE - primarily through the date.
        Content should read naturally for its time period without screaming "outdated".
        """
        temporal_phrases = {
            DocumentFormat.MD: [
                "Write the document as if it were current at the time of writing.",
                "Use present tense to describe the information as valid.",
                "The document should read as authoritative for its time period.",
                "Include natural date references that match the document date.",
            ],
            DocumentFormat.EMAIL: [
                "Write the email naturally as if sent on that date.",
                "The sender should convey the information confidently.",
                "Use normal business communication style for the time.",
                "Include a natural email signature with the date context.",
            ],
            DocumentFormat.CHAT: [
                "Write the chat as a natural conversation from that time.",
                "Participants should discuss the topic as current information.",
                "Use casual language appropriate for workplace chat.",
                "Include natural timestamp references.",
            ],
        }
        
        phrases = temporal_phrases.get(doc_format, temporal_phrases[DocumentFormat.MD])
        selected = random.sample(phrases, min(2, len(phrases)))
        
        base_context = f"Document date: {formatted_date}. "
        base_context += (
            "Write this as the current state of affairs as of the document date above — "
            "the author believes this information is accurate right now. "
            "Do NOT reference it as historical, archived, or outdated. "
            "Do NOT use phrases like 'back in', 'previously', or 'as of [earlier date]'. "
            "The date field is the only staleness signal; the content must read as authoritative for its time."
        )
        base_context += " " + " ".join(selected)
        
        return base_context
    
    def _generate_temporal_answer(
        self, fact: Fact, doc_date: Optional[str] = None
    ) -> Tuple[str, str]:
        """Generate what the fact's value was as of doc_date (before the current gold)."""
        date_hint = (
            f"The document that will contain this answer is dated {doc_date}. "
            "Generate a value that would have been accurate and considered current as of that date."
            if doc_date
            else "Generate a value from 1-3 years before the current answer."
        )
        prompt = f"""You are writing a fact for a document dated {doc_date or 'in the past'}.
The document will present this fact as its CURRENT finding — not as a historical reference.

CURRENT (most recent) answer: {fact.answer}
QUESTION: {fact.question}

{date_hint}

STRICT RULES:
- The time period in your answer MUST differ from the time period in the current answer.
  e.g. if current says "Q2 2024", your answer must say "Q1 2023" or "Q4 2022", NOT "Q2 2024".
- Change the numeric value OR the referenced date/period — both if possible.
- Do NOT use the phrase "as of [same date as current answer]".
- The value can be higher or lower than current — both directions are valid.

Return JSON:
{{"temporal_answer": "The value as it stood on {doc_date or 'the document date'}", "reason": "Brief note on what period this covers"}}"""

        response = prompt_llm(prompt, model=self.model)
        try:
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
            data = json.loads(response)
            return data.get("temporal_answer", fact.answer), data.get("reason", "Historical value, since updated")
        except:
            return f"[Historical] {fact.answer}", "Historical data, since changed"
    
    
    def generate_noise_document(
        self,
        distractor_fact: Fact,
        question: str,
        company_info: Dict,
        persona: Dict,
        output_dir: str,
        reference_date: Optional[datetime] = None,
    ) -> Document:
        """
        Generate a NOISE document from a distractor fact.
        
        Noise: Contains CORRECT information that is IRRELEVANT to the question.
        These are valid, accurate documents about unrelated topics that serve as
        distractors. The information itself is true but doesn't help answer the
        target question.
        """
        doc_format = select_format_by_probability(DocumentType.NOISE)
        logger.info(f"  Format: {doc_format.value}")
        
        # Generate a varied date for noise documents
        doc_date = generate_document_date(DocumentType.NOISE, reference_date)
        formatted_date = format_date_for_document(doc_date, doc_format)
        
        fact_dict = self._create_fact_dict_for_generator(
            distractor_fact,
            additional_context=f"Document date: {formatted_date}. "
                             "This is correct information about a different topic. "
                             "Present this as legitimate documentation about an unrelated subject."
        )
        
        text, file_path = self._generate_with_file_generator(
            doc_format, fact_dict, company_info, persona, output_dir
        )
        
        return Document(
            text=text,
            doc_type=DocumentType.NOISE.value,
            answer=distractor_fact.answer,  # The noise doc has its own correct answer
            doc_format=doc_format.value,
            document_date=doc_date,
            source_fact=distractor_fact.to_dict(),
            reliability_signals={
                "recency": "current",
                "verification": "verified",
                "authority": "high",
                "relevance": "off_topic",  # Correct info, but irrelevant to target question
            },
            file_path=file_path,
            confidence=sample_confidence(DocumentType.NOISE),
        )
    
    def generate_conflict_task(
        self,
        question: str,
        ground_truth_facts: List[Fact],
        distractor_facts: List[Fact],
        company_info: Dict,
        persona: Dict,
        output_dir: str,
        doc_counts: Optional[Dict] = None,
        num_correct: int = 1,
        num_misinfo: int = 1,
        num_temporal: int = 1,
        num_noise: int = 2,
        max_workers: int = 5,
        reference_date: Optional[datetime] = None,
    ) -> ConflictTask:
        """
        Generate a complete conflict task with mixed document types.

        Supports multi-answer mode: when doc_counts is provided, each fact in
        ground_truth_facts gets its own set of correct/misinfo/temporal documents.

        Args:
            ground_truth_facts: List of Fact objects (one per answer)
            doc_counts: Per-answer document counts dict with "per_answer" and "num_noise"
                        keys. When None, falls back to flat count params (legacy mode).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        documents = []
        gold_answers = []
        wrong_answers = []

        # Use current time as reference if not provided
        if reference_date is None:
            reference_date = datetime.now()

        task_output_dir = os.path.join(output_dir, "generated_files")
        os.makedirs(task_output_dir, exist_ok=True)

        available_distractors = distractor_facts.copy()
        random.shuffle(available_distractors)

        # Build tasks for parallel execution
        tasks = []

        if doc_counts is not None:
            # Multi-answer mode: per-answer document counts
            for i, fact in enumerate(ground_truth_facts):
                answer_counts = doc_counts["per_answer"][i]
                for _ in range(answer_counts["num_correct"]):
                    tasks.append(("correct", fact, None))
                    gold_answers.append(fact.answer)
                for _ in range(answer_counts["num_misinfo"]):
                    tasks.append(("misinfo", fact, None))
                for _ in range(answer_counts["num_temporal"]):
                    tasks.append(("temporal", fact, None))
            noise_count = min(doc_counts["num_noise"], len(available_distractors))
            for i in range(noise_count):
                tasks.append(("noise", available_distractors[i], question))
        else:
            # Legacy single-fact mode
            ground_truth_fact = ground_truth_facts[0]
            for i in range(num_correct):
                tasks.append(("correct", ground_truth_fact, None))
                gold_answers.append(ground_truth_fact.answer)
            for i in range(num_misinfo):
                tasks.append(("misinfo", ground_truth_fact, None))
            for i in range(num_temporal):
                tasks.append(("temporal", ground_truth_fact, None))
            for i in range(min(num_noise, len(available_distractors))):
                fact = available_distractors[i]
                tasks.append(("noise", fact, question))
        
        doc_pbar = tqdm(total=len(tasks), desc="  Generating documents", unit="doc", leave=False)
        
        def process_task(task_info):
            doc_type, fact, extra = task_info
            
            if doc_type == "correct":
                return ("correct", self.generate_correct_document(
                    fact, company_info, persona, task_output_dir, reference_date
                ), None)
            elif doc_type == "misinfo":
                doc, wrong_answer = self.generate_misinfo_document(
                    fact, company_info, persona, task_output_dir, reference_date
                )
                return ("misinfo", doc, wrong_answer)
            elif doc_type == "temporal":
                doc, temporal_answer = self.generate_temporal_document(
                    fact, company_info, persona, task_output_dir, reference_date
                )
                return ("temporal", doc, temporal_answer)
            elif doc_type == "noise":
                return ("noise", self.generate_noise_document(
                    fact, extra, company_info, persona, task_output_dir, reference_date
                ), None)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_task, task): task for task in tasks}
            
            for future in as_completed(futures):
                task_info = futures[future]
                try:
                    doc_type, doc, extra_result = future.result()
                    documents.append(doc)
                    
                    # Track wrong answers from misinfo and temporal documents
                    if doc_type in ["misinfo", "temporal"] and extra_result:
                        wrong_answers.append(extra_result)
                    
                    doc_pbar.set_postfix({"type": doc_type, "fact": task_info[1].fact_id[:15]})
                    doc_pbar.update(1)
                except Exception as e:
                    tqdm.write(f"Error generating {task_info[0]} document: {e}")
                    doc_pbar.update(1)
        
        doc_pbar.close()
        
        random.shuffle(documents)
        
        doc_dicts = []
        for doc in documents:
            doc_dict = {
                "text": doc.text,
                "type": doc.doc_type,
                "answer": doc.answer,
                "format": doc.doc_format,
                "document_date": doc.document_date,  # Include date in output
                "source_fact": doc.source_fact,
            }
            if doc.confidence is not None:
                doc_dict["confidence"] = doc.confidence
            if doc.reliability_signals:
                doc_dict["reliability_signals"] = doc.reliability_signals
            if doc.file_path:
                doc_dict["file_path"] = doc.file_path
            doc_dicts.append(doc_dict)
        
        company_name = company_info.get("name", "Enterprise")
        
        return ConflictTask(
            question=question,
            documents=doc_dicts,
            disambig_entity=[company_name],
            gold_answers=list(set(gold_answers)),
            wrong_answers=list(set(wrong_answers)),
        )


def get_task_config_path(facts_csv_path: str, tasks_dir: str = None) -> Optional[str]:
    """
    Get the task config path for a given facts CSV file.
    
    Args:
        facts_csv_path: Path to facts CSV file (e.g., DR0001_facts.csv)
        tasks_dir: Directory containing task config files
    
    Returns:
        Path to task config file, or None if not found
    """
    csv_path = Path(facts_csv_path)
    task_id = csv_path.stem.replace("_facts", "")
    
    # If tasks_dir is provided, look there
    if tasks_dir:
        config_path = Path(tasks_dir) / f"{task_id}.json"
        if config_path.exists():
            return str(config_path)
    
    # Look in default data/tasks directory relative to script
    script_dir = Path(__file__).parent
    default_tasks_dir = script_dir / "data" / "tasks"
    config_path = default_tasks_dir / f"{task_id}.json"
    if config_path.exists():
        return str(config_path)
    
    # Look in sample_data directory
    sample_config = script_dir / "sample_data" / "task.json"
    if sample_config.exists():
        return str(sample_config)
    
    return None


def load_task_config(config_path: str) -> Tuple[Dict, Dict]:
    """
    Load company_info and persona from a task config JSON file.
    
    Args:
        config_path: Path to task.json config file
    
    Returns:
        Tuple of (company_info_dict, persona_dict)
    """
    default_company_info = {
        "name": "Enterprise Corp",
        "industry": "Technology",
        "description": "A technology company",
    }
    
    default_persona = {
        "first_name": "Alex",
        "last_name": "Johnson",
        "email": "alex.johnson@company.com",
        "role": "Senior Analyst",
        "department": "Strategy",
        "responsibilities": "Data analysis and reporting",
    }
    
    if not config_path or not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return default_company_info, default_persona
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            task_config = json.load(f)
        
        company_info = task_config.get("company_info", default_company_info)
        persona = task_config.get("persona", default_persona)
        
        logger.info(f"Loaded task config from {config_path}")
        logger.info(f"  Company: {company_info.get('name', 'Unknown')}")
        logger.info(f"  Persona: {persona.get('name', persona.get('first_name', 'Unknown'))}")
        
        return company_info, persona
        
    except Exception as e:
        logger.warning(f"Error loading config: {e}, using defaults")
        return default_company_info, default_persona


def main():
    parser = argparse.ArgumentParser(
        description="Generate RAMDocs-style conflict dataset"
    )
    parser.add_argument(
        "--facts-csv",
        type=str,
        help="Path to facts CSV file",
    )
    parser.add_argument(
        "--facts-dir",
        type=str,
        default=None,
        help="Directory containing facts CSV files (processes all *_facts.csv)",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default=None,
        help="Path to task.json config file (if not provided, auto-discovers from data/tasks/)",
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default=None,
        help="Directory containing task config files (e.g., DR0001.json, DR0002.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/conflict_dataset",
        help="Output directory for generated dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--num-correct",
        type=int,
        default=1,
        help="Number of correct documents per task",
    )
    parser.add_argument(
        "--num-misinfo",
        type=int,
        default=1,
        help="Number of misinformation documents per task (with uncertainty markers)",
    )
    parser.add_argument(
        "--num-temporal",
        type=int,
        default=1,
        help="Number of temporal conflict documents per task (outdated/superseded info)",
    )
    parser.add_argument(
        "--num-noise",
        type=int,
        default=2,
        help="Number of noise documents per task (correct but irrelevant info)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format (jsonl for one task per line, json for array)",
    )
    parser.add_argument(
        "--random-counts",
        action="store_true",
        default=False,
        help="Use random document counts (5-10 total) instead of fixed counts",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of parallel API workers (default: 5)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of task examples to generate (for testing/sampling)",
    )
    parser.add_argument(
        "--num-correct-per-answer",
        type=int,
        default=2,
        help="Number of correct documents per answer in multi-answer mode (default: 2)",
    )
    parser.add_argument(
        "--num-misinfo-per-answer",
        type=int,
        default=1,
        help="Number of misinfo documents per answer in multi-answer mode (default: 1)",
    )
    parser.add_argument(
        "--num-temporal-per-answer",
        type=int,
        default=1,
        help="Number of temporal documents per answer in multi-answer mode (default: 1)",
    )

    args = parser.parse_args()
    
    # Collect CSV files to process
    csv_files = []
    script_dir = Path(__file__).parent
    
    if args.facts_csv:
        csv_files.append(Path(args.facts_csv))
    elif args.facts_dir:
        facts_dir = Path(args.facts_dir)
        csv_files = list(facts_dir.glob("*_facts.csv"))
    else:
        # Check for data/facts directory (default location)
        data_dir = script_dir / "data" / "facts"
        if data_dir.exists():
            csv_files = list(data_dir.glob("*_facts.csv"))
        
        # Fallback to sample_data directory
        if not csv_files:
            sample_dir = script_dir / "sample_data"
            if sample_dir.exists():
                csv_files = list(sample_dir.glob("*_facts.csv"))
        
        if not csv_files:
            logger.error("No facts files specified. Use --facts-csv or --facts-dir")
            logger.error("Or place facts files in data/facts/ directory")
            return
    
    if not csv_files:
        logger.error("No facts CSV files found")
        return
    
    # Sort files by name for consistent ordering
    csv_files = sorted(csv_files, key=lambda x: x.name)
    logger.info(f"Found {len(csv_files)} facts files to process")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = ConflictDatasetGenerator(model=args.model)
    
    # Process each facts file
    all_tasks = []
    
    for csv_file in tqdm(csv_files, desc="Processing facts files", unit="file"):
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Processing: {csv_file}")
        tqdm.write(f"{'='*60}")
        
        try:
            dr_question, supporting_facts, distractor_facts = generator.load_facts_from_csv(str(csv_file))
            
            if not supporting_facts:
                logger.warning(f"No supporting facts found in {csv_file}, skipping")
                continue
            
            tqdm.write(f"DR Question: {dr_question[:100]}...")
            tqdm.write(f"Supporting facts: {len(supporting_facts)}")
            tqdm.write(f"Distractor facts: {len(distractor_facts)}")
            
            # Load task config for this specific facts file
            if args.task_config:
                config_path = args.task_config
            else:
                config_path = get_task_config_path(str(csv_file), args.tasks_dir)
            
            company_info, persona = load_task_config(config_path)
            
            # One task per supporting fact (single-answer mode).
            # Each task still has multiple documents per answer (correct + misinfo + temporal).
            for fact_idx, supporting_fact in enumerate(supporting_facts):
                tqdm.write(f"\n  Fact {fact_idx + 1}/{len(supporting_facts)}: {supporting_fact.question[:60]}...")

                if args.random_counts:
                    doc_counts = generate_random_document_counts_multi(num_answers=1)
                    tqdm.write(f"    Random counts: {doc_counts}")
                else:
                    doc_counts = {
                        "per_answer": [
                            {
                                "num_correct": args.num_correct_per_answer,
                                "num_misinfo": args.num_misinfo_per_answer,
                                "num_temporal": args.num_temporal_per_answer,
                            }
                        ],
                        "num_noise": args.num_noise,
                    }

                task = generator.generate_conflict_task(
                    question=supporting_fact.question,
                    ground_truth_facts=[supporting_fact],
                    distractor_facts=distractor_facts,
                    company_info=company_info,
                    persona=persona,
                    output_dir=str(output_dir),
                    doc_counts=doc_counts,
                    max_workers=args.max_workers,
                )

                all_tasks.append(task.to_dict())

                doc_types = {}
                doc_formats = {}
                for doc in task.documents:
                    doc_type = doc["type"]
                    doc_format = doc["format"]
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    doc_formats[doc_format] = doc_formats.get(doc_format, 0) + 1

                logger.info(f"  Generated task with {len(task.documents)} documents:")
                logger.info(f"    By type: {doc_types}")
                logger.info(f"    Gold answer: {task.gold_answers[0][:60] if task.gold_answers else 'N/A'}")
                logger.info(f"    Wrong answers: {len(task.wrong_answers)}")

                if args.max_examples and len(all_tasks) >= args.max_examples:
                    break

            if args.max_examples and len(all_tasks) >= args.max_examples:
                tqdm.write(f"\nReached --max-examples limit ({args.max_examples}), stopping.")
                break
            
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save output
    if not all_tasks:
        logger.error("No tasks generated")
        return
        
    if args.output_format == "jsonl":
        output_file = output_dir / "conflict_dataset.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for task in all_tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
    else:
        output_file = output_dir / "conflict_dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_tasks, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset generation complete!")
    logger.info(f"Total tasks: {len(all_tasks)}")
    logger.info(f"Output saved to: {output_file}")
    logger.info(f"{'='*60}")
    
    # Print statistics
    total_docs = sum(len(t["documents"]) for t in all_tasks)
    if total_docs == 0:
        logger.warning("No documents generated")
        return
        
    total_correct = sum(sum(1 for d in t["documents"] if d["type"] == "correct") for t in all_tasks)
    total_misinfo = sum(sum(1 for d in t["documents"] if d["type"] == "misinfo") for t in all_tasks)
    total_temporal = sum(sum(1 for d in t["documents"] if d["type"] == "temporal") for t in all_tasks)
    total_noise = sum(sum(1 for d in t["documents"] if d["type"] == "noise") for t in all_tasks)
    
    format_counts = {}
    for task in all_tasks:
        for doc in task["documents"]:
            fmt = doc.get("format", "unknown")
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
    
    print(f"\nDataset Statistics:")
    print(f"  Total documents: {total_docs}")
    print(f"  By type:")
    print(f"    - Correct: {total_correct} ({100*total_correct/total_docs:.1f}%) - Accurate, verified information")
    print(f"    - Misinfo: {total_misinfo} ({100*total_misinfo/total_docs:.1f}%) - Misinformation with uncertainty markers")
    print(f"    - Temporal: {total_temporal} ({100*total_temporal/total_docs:.1f}%) - Outdated/superseded information")
    print(f"    - Noise: {total_noise} ({100*total_noise/total_docs:.1f}%) - Correct but irrelevant information")
    print(f"  By format:")
    for fmt, count in sorted(format_counts.items()):
        print(f"    - {fmt}: {count} ({100*count/total_docs:.1f}%)")
    print(f"  Avg docs per task: {total_docs/len(all_tasks):.1f}")


if __name__ == "__main__":
    main()

