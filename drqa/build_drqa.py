#!/usr/bin/env python3
"""
Generate RAMDocs-style conflict dataset from new DRBench task directory structure.

Reads from data/tasks/DR*/ (each containing context.json, dr_question.json,
files/IN*/qa_dict.json) instead of the old CSV-based facts.

Existing DRBench-generated files are reused directly:
  - correct   (1 per task, from the IN_ file — no extra LLM correct docs)
  - noise     (DI_ files, no LLM calls)

LLM generation is reserved for:
  - misinfo   (plausible wrong answer with uncertainty markers)
  - temporal  (outdated answer, signalled by older date)

After generation, an LLM-as-judge verification step filters low-quality examples
using type-specific rubrics (correct entailment, misinfo plausibility/distinctness,
temporal appropriateness, noise off-topicness).

Output format matches dr_factual_conflict_v2.jsonl.
"""

import argparse
import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from conflict_generator import (
    ConflictDatasetGenerator,
    Document,
    DocumentType,
    Fact,
    generate_document_date,
    generate_random_document_counts_multi,
    sample_confidence,
)
from llm import DEFAULT_MODEL, prompt_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verification prompts (fixed rubrics for reproducibility)
# ---------------------------------------------------------------------------

_VERIFY_CORRECT = """\
You are quality-checking a "correct" document in a factual QA conflict benchmark.

Question: {question}
Gold answer: {gold_answer}
Document (excerpt):
{text}

Does this document explicitly state or clearly entail the gold answer?

Respond with valid JSON only — no markdown, no prose outside the object:
{{
  "verdict": "PASS" or "FAIL",
  "reason": "<one sentence>",
  "confidence": <float 0.0–1.0>
}}

PASS: the document contains the gold answer verbatim or as an unambiguous paraphrase.
FAIL: the gold answer cannot be inferred from this document alone."""

_VERIFY_MISINFO = """\
You are quality-checking a "misinformation" document in a factual QA conflict benchmark.

Question: {question}
Gold answer (correct): {gold_answer}
Claimed answer in this document: {claimed_answer}
Document (excerpt):
{text}

Verify two properties:
(A) DISTINCT — the claimed answer is factually and specifically different from the gold answer.
(B) PLAUSIBLE — the claimed answer sounds credible to a naive reader (not obviously absurd).

Respond with valid JSON only:
{{
  "verdict": "PASS" or "FAIL",
  "distinct": <true/false>,
  "plausible": <true/false>,
  "reason": "<one sentence>",
  "confidence": <float 0.0–1.0>
}}

PASS requires distinct=true AND plausible=true."""

_VERIFY_TEMPORAL = """\
You are quality-checking a "temporal conflict" document in a factual QA conflict benchmark.

Question: {question}
Current gold answer: {gold_answer}
Historical answer in this document: {claimed_answer}
Document date: {document_date}
Document (excerpt):
{text}

Verify two properties:
(A) DISTINCT — the historical answer differs meaningfully from the current gold answer (different number,
    date, or factual claim). Do NOT judge whether the direction of change is plausible — metrics can
    improve or worsen over time for many valid business reasons.
(B) NOT_LABELLED — the document does NOT contain explicit "archived", "outdated", or "superseded" labels.
    The document date is the only staleness signal; the text should read as authoritative for its time.

Respond with valid JSON only:
{{
  "verdict": "PASS" or "FAIL",
  "distinct": <true/false>,
  "not_labelled": <true/false>,
  "reason": "<one sentence>",
  "confidence": <float 0.0–1.0>
}}

PASS requires distinct=true AND not_labelled=true."""

_VERIFY_NOISE = """\
You are quality-checking a "noise" document in a factual QA conflict benchmark.

Question: {question}
Gold answer (for reference): {gold_answer}
Document (excerpt):
{text}

Does this document NOT answer the target question? (i.e. is it genuinely about a different topic and does not contain or clearly imply the gold answer?)

Respond with valid JSON only:
{{
  "verdict": "PASS" or "FAIL",
  "reason": "<one sentence>",
  "confidence": <float 0.0–1.0>
}}

FAIL if the document contains the gold answer or makes a strongly similar claim.
PASS if the document is about a different topic entirely."""

_TEXT_LIMIT = 2000  # characters sent to verification LLM (misinfo / temporal / noise)


def _parse_verify_response(response: str) -> dict:
    """Extract JSON from LLM verification response (strips any markdown fences)."""
    text = response.strip()
    # Strip ```json ... ``` fences if present
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return json.loads(text)


def verify_document(
    doc: Dict,
    question: str,
    gold_answer: str,
    model: str,
) -> Dict:
    """
    Run LLM-as-judge verification on a single document.

    Correct documents from existing DRBench files are auto-passed: they were
    produced by DRBench's needle-in-a-haystack pipeline and spot-checked by
    human annotators, so re-verifying them with a truncated excerpt would only
    produce false negatives (the answer is buried deep in a long document).

    LLM verification is applied to misinfo, temporal, and noise docs, and to
    any correct doc that was LLM-generated (the rare pptx-fallback case).

    Returns a dict with at minimum: verdict ("PASS"/"FAIL"), reason, confidence.
    Additional type-specific fields (distinct, plausible, …) are included when present.
    """
    doc_type = doc["type"]

    # Auto-pass existing DRBench correct documents — human-verified at source.
    # Generated-files path indicates an LLM fallback; verify those normally.
    if doc_type == "correct":
        file_path = doc.get("file_path", "")
        if "generated_files" not in str(file_path):
            return {
                "verdict": "PASS",
                "reason": "existing DRBench file, human-verified at source",
                "confidence": 1.0,
            }

    text_excerpt = doc["text"][:_TEXT_LIMIT]

    if doc_type == "correct":
        prompt = _VERIFY_CORRECT.format(
            question=question,
            gold_answer=gold_answer,
            text=text_excerpt,
        )
    elif doc_type == "misinfo":
        prompt = _VERIFY_MISINFO.format(
            question=question,
            gold_answer=gold_answer,
            claimed_answer=doc.get("answer", ""),
            text=text_excerpt,
        )
    elif doc_type == "temporal":
        prompt = _VERIFY_TEMPORAL.format(
            question=question,
            gold_answer=gold_answer,
            claimed_answer=doc.get("answer", ""),
            document_date=doc.get("document_date", "unknown"),
            text=text_excerpt,
        )
    elif doc_type == "noise":
        prompt = _VERIFY_NOISE.format(
            question=question,
            gold_answer=gold_answer,
            text=text_excerpt,
        )
    else:
        return {"verdict": "PASS", "reason": "unknown type skipped", "confidence": 1.0}

    try:
        raw = prompt_llm(prompt, model=model)
        result = _parse_verify_response(raw)
        if "verdict" not in result:
            result["verdict"] = "FAIL"
            result.setdefault("reason", "malformed response")
        return result
    except Exception as e:
        logger.warning(f"Verification call failed ({doc_type}): {e}")
        return {"verdict": "FAIL", "reason": f"verification error: {e}", "confidence": 0.0}


def verify_task(
    task_dict: Dict,
    model: str,
    max_workers: int = 5,
) -> Tuple[bool, List[Dict]]:
    """
    Verify all documents in a task in parallel.

    Returns:
        (all_passed: bool, per_doc_results: list of dicts with doc index + verdict info)
    """
    question = task_dict["question"]
    gold_answer = task_dict["gold_answers"][0] if task_dict["gold_answers"] else ""
    documents = task_dict["documents"]

    results = [None] * len(documents)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(verify_document, doc, question, gold_answer, model): i
            for i, doc in enumerate(documents)
        }
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            result["doc_index"] = idx
            result["doc_type"] = documents[idx]["type"]
            results[idx] = result

    all_passed = all(r["verdict"] == "PASS" for r in results)
    return all_passed, results


# ---------------------------------------------------------------------------
# Reading existing DRBench files
# ---------------------------------------------------------------------------

def _render_pptx_as_markdown(fd: dict) -> str:
    """Reconstruct a pptx document as Markdown from its file_dict.json."""
    lines = [f"# {fd.get('file_title', 'Presentation')}", ""]
    intro = fd.get("introduction", "").strip()
    if intro:
        lines += ["## Introduction", "", intro, ""]
    for section in fd.get("subsections", []):
        heading = section.get("heading", "").strip()
        content = section.get("content", "").strip()
        if heading:
            lines += [f"## {heading}", ""]
        if content:
            lines += [content, ""]
    conclusion = fd.get("conclusion", "").strip()
    if conclusion:
        lines += ["## Conclusion", "", conclusion, ""]
    return "\n".join(lines)


def _read_existing_file(fact_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (text, format_str) by reading the best available file in fact_dir.

    Format mapping:
      pdf / docx      → .md file        → "md"
      pptx            → file_dict.json  → "md"  (rendered as Markdown)
      xlsx            → .csv file       → "md"  (tabular text)
      jsonl + email   → .jsonl          → "email"
      jsonl + mattermost → .jsonl       → "chat"
    """
    fd_path = fact_dir / "file_dict.json"
    if not fd_path.exists():
        return None, None

    fd = json.loads(fd_path.read_text(encoding="utf-8"))
    file_format = fd.get("file_format", "")
    app = fd.get("app", "")

    if file_format == "jsonl":
        fmt = "chat" if app == "mattermost" else "email"
        jsonl_files = [f for f in fact_dir.iterdir() if f.suffix == ".jsonl"]
        if jsonl_files:
            return jsonl_files[0].read_text(encoding="utf-8"), fmt
        return None, None

    md_files = [f for f in fact_dir.iterdir() if f.suffix == ".md"]
    if md_files:
        return md_files[0].read_text(encoding="utf-8"), "md"

    # pptx: reconstruct from file_dict.json structure
    if file_format == "pptx" and "file_title" in fd:
        return _render_pptx_as_markdown(fd), "md"

    csv_files = [f for f in fact_dir.iterdir() if f.suffix == ".csv"]
    if csv_files:
        return csv_files[0].read_text(encoding="utf-8"), "md"

    return None, None


def _make_doc_from_existing(
    fact: Fact,
    fact_dir: Path,
    doc_type: DocumentType,
) -> Optional[Document]:
    """Create a Document from an existing file, or return None if unavailable."""
    text, fmt = _read_existing_file(fact_dir)
    if text is None:
        return None

    doc_date = generate_document_date(doc_type)

    if doc_type == DocumentType.CORRECT:
        signals = {"recency": "current", "verification": "verified", "authority": "high"}
    else:  # NOISE
        signals = {
            "recency": "current",
            "verification": "verified",
            "authority": "high",
            "relevance": "off_topic",
        }

    return Document(
        text=text,
        doc_type=doc_type.value,
        answer=fact.answer,
        doc_format=fmt,
        document_date=doc_date,
        source_fact=fact.to_dict(),
        reliability_signals=signals,
        file_path=str(fact_dir),
        confidence=sample_confidence(doc_type),
    )


# ---------------------------------------------------------------------------
# Loading facts from a task directory
# ---------------------------------------------------------------------------

def load_facts_from_task_dir(
    task_dir: Path,
) -> Tuple[str, List[Tuple[Fact, Path]], List[Tuple[Fact, Path]], Dict, Dict]:
    """
    Load facts from a DRBench task directory.

    Returns:
        (dr_question,
         supporting_facts_with_dirs,   # List[(Fact, Path)]
         distractor_facts_with_dirs,   # List[(Fact, Path)]
         company_info, persona)
    """
    context = json.loads((task_dir / "context.json").read_text(encoding="utf-8"))
    company_info = context["company_info"]
    persona = context["persona"]

    dr_q = json.loads((task_dir / "dr_question.json").read_text(encoding="utf-8"))
    dr_question = dr_q["dr_question"]

    supporting: List[Tuple[Fact, Path]] = []
    distractor: List[Tuple[Fact, Path]] = []
    seen_distractor_qa: set = set()

    files_dir = task_dir / "files"
    if not files_dir.exists():
        return dr_question, supporting, distractor, company_info, persona

    for fact_dir in sorted(files_dir.iterdir()):
        if not fact_dir.is_dir():
            continue
        qa_path = fact_dir / "qa_dict.json"
        if not qa_path.exists():
            continue

        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        fact_id = fact_dir.name

        # Use qa_type from qa_dict.json as the authoritative role signal.
        # Directory naming conventions (IN_ / DI_ prefix, qa_XXXXXXXXXX hash)
        # are unreliable across task generations — qa_type is always present.
        is_supporting = qa.get("qa_type") == "insight"

        question = qa["specific_question"]
        answer = qa["answer"]

        # Deduplicate distractor entries that share Q&A across file formats
        if not is_supporting:
            key = (question, answer)
            if key in seen_distractor_qa:
                continue
            seen_distractor_qa.add(key)

        fact = Fact(
            fact_id=fact_id,
            fact_type="supporting" if is_supporting else "distractor",
            source="internal",
            question=question,
            answer=answer,
            dr_question=dr_question,
        )

        if is_supporting:
            supporting.append((fact, fact_dir))
        else:
            distractor.append((fact, fact_dir))

    return dr_question, supporting, distractor, company_info, persona


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def generate_task(
    generator: ConflictDatasetGenerator,
    question: str,
    supporting_fact: Fact,
    supporting_fact_dir: Path,
    distractor_facts_with_dirs: List[Tuple[Fact, Path]],
    company_info: Dict,
    persona: Dict,
    output_dir: str,
    num_misinfo: int = 1,
    num_temporal: int = 1,
    num_noise: int = 2,
    max_workers: int = 5,
) -> Dict:
    """
    Generate one conflict task.

    Correct doc:  1, always from the existing IN_ file (LLM fallback only if
                  the file format has no readable text, e.g. pptx-only).
    Noise docs:   from existing DI_ files; LLM fallback only if no readable file.
    Misinfo docs: always LLM (needs a plausible wrong answer).
    Temporal docs: always LLM (needs an outdated answer).
    """
    task_output_dir = Path(output_dir) / "generated_files"
    task_output_dir.mkdir(parents=True, exist_ok=True)

    gold_answers = [supporting_fact.answer]
    wrong_answers: List[str] = []
    documents: List[Document] = []

    # --- correct doc (1, from existing file) ---
    existing_correct = _make_doc_from_existing(
        supporting_fact, supporting_fact_dir, DocumentType.CORRECT
    )
    if existing_correct is not None:
        documents.append(existing_correct)
        llm_correct_needed = 0
    else:
        # pptx-only or missing file — generate one via LLM as fallback
        llm_correct_needed = 1

    # --- noise docs (from existing DI_ files) ---
    available_distractors = list(distractor_facts_with_dirs)
    random.shuffle(available_distractors)
    llm_noise_facts: List[Fact] = []
    for dfact, ddir in available_distractors[:num_noise]:
        doc = _make_doc_from_existing(dfact, ddir, DocumentType.NOISE)
        if doc is not None:
            documents.append(doc)
        else:
            llm_noise_facts.append(dfact)

    # --- LLM tasks (misinfo + temporal + fallbacks) ---
    llm_tasks = []
    for _ in range(llm_correct_needed):
        llm_tasks.append(("correct", supporting_fact))
    for _ in range(num_misinfo):
        llm_tasks.append(("misinfo", supporting_fact))
    for _ in range(num_temporal):
        llm_tasks.append(("temporal", supporting_fact))
    for f in llm_noise_facts:
        llm_tasks.append(("noise", f))

    if llm_tasks:
        def _run(task_info):
            dtype, fact = task_info
            if dtype == "correct":
                doc = generator.generate_correct_document(
                    fact, company_info, persona, str(task_output_dir)
                )
                return dtype, doc, fact.answer
            elif dtype == "misinfo":
                doc, wrong = generator.generate_misinfo_document(
                    fact, company_info, persona, str(task_output_dir)
                )
                return dtype, doc, wrong
            elif dtype == "temporal":
                doc, old = generator.generate_temporal_document(
                    fact, company_info, persona, str(task_output_dir)
                )
                return dtype, doc, old
            else:
                doc = generator.generate_noise_document(
                    fact, question, company_info, persona, str(task_output_dir)
                )
                return dtype, doc, None

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_run, t): t for t in llm_tasks}
            for future in as_completed(futures):
                try:
                    dtype, doc, extra = future.result()
                    documents.append(doc)
                    if dtype == "correct":
                        gold_answers.append(extra)
                    elif dtype in ("misinfo", "temporal") and extra:
                        wrong_answers.append(extra)
                except Exception as e:
                    logger.error(f"LLM doc generation failed: {e}")

    random.shuffle(documents)

    doc_dicts = [
        {
            "text": doc.text,
            "type": doc.doc_type,
            "answer": doc.answer,
            "format": doc.doc_format,
            "document_date": doc.document_date,
            "source_fact": doc.source_fact,
            "confidence": doc.confidence,
            "reliability_signals": doc.reliability_signals,
            "file_path": doc.file_path,
        }
        for doc in documents
    ]

    gold = list(dict.fromkeys(gold_answers))
    wrong = list(set(wrong_answers))
    company_name = company_info.get("name", "Enterprise")

    return {
        "question": question,
        "documents": doc_dicts,
        "disambig_entity": [company_name],
        "gold_answers": gold,
        "wrong_answers": wrong,
        "gold_answers_full": gold,
        "wrong_answers_full": wrong,
    }


# ---------------------------------------------------------------------------
# Per-fact worker (top-level function so it can run in a thread pool)
# ---------------------------------------------------------------------------

def _regenerate_failed_docs(
    task_dict: Dict,
    verify_results: List[Dict],
    generator: ConflictDatasetGenerator,
    fact: Fact,
    fact_dir: Path,
    distractor_facts_with_dirs: List[Tuple[Fact, Path]],
    company_info: Dict,
    persona: Dict,
    output_dir: str,
    nn: int,
) -> Dict:
    """
    Replace only the documents that failed verification, keeping passing ones.
    Regenerates misinfo, temporal, and noise docs as needed.
    """
    task_output_dir = Path(output_dir) / "generated_files"
    task_output_dir.mkdir(parents=True, exist_ok=True)

    failed_types = {r["doc_type"] for r in verify_results if r["verdict"] == "FAIL"}

    # Keep all passing documents
    kept_docs = [
        doc for doc, r in zip(task_dict["documents"], verify_results)
        if r["verdict"] == "PASS"
    ]

    new_docs = list(kept_docs)
    gold_answers = list(task_dict["gold_answers"])
    wrong_answers = list(task_dict["wrong_answers"])

    llm_tasks = []
    if "misinfo" in failed_types:
        llm_tasks.append(("misinfo", fact))
    if "temporal" in failed_types:
        llm_tasks.append(("temporal", fact))
    if "noise" in failed_types:
        # Pick a different distractor for noise
        available = [
            (f, d) for f, d in distractor_facts_with_dirs
            if not any(
                doc.get("source_fact", {}).get("fact_id") == f.fact_id
                for doc in kept_docs
            )
        ]
        random.shuffle(available)
        for dfact, ddir in available[:nn]:
            doc = _make_doc_from_existing(dfact, ddir, DocumentType.NOISE)
            if doc:
                new_docs.append({
                    "text": doc.text, "type": doc.doc_type, "answer": doc.answer,
                    "format": doc.doc_format, "document_date": doc.document_date,
                    "source_fact": doc.source_fact, "confidence": doc.confidence,
                    "reliability_signals": doc.reliability_signals,
                    "file_path": doc.file_path,
                })

    def _run(task_info):
        dtype, f = task_info
        if dtype == "misinfo":
            doc, wrong = generator.generate_misinfo_document(
                f, company_info, persona, str(task_output_dir)
            )
            return dtype, doc, wrong
        elif dtype == "temporal":
            doc, old = generator.generate_temporal_document(
                f, company_info, persona, str(task_output_dir)
            )
            return dtype, doc, old

    if llm_tasks:
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {ex.submit(_run, t): t for t in llm_tasks}
            for future in as_completed(futures):
                try:
                    dtype, doc, extra = future.result()
                    new_docs.append({
                        "text": doc.text, "type": doc.doc_type, "answer": doc.answer,
                        "format": doc.doc_format, "document_date": doc.document_date,
                        "source_fact": doc.source_fact, "confidence": doc.confidence,
                        "reliability_signals": doc.reliability_signals,
                        "file_path": doc.file_path,
                    })
                    if dtype in ("misinfo", "temporal") and extra:
                        wrong_answers.append(extra)
                except Exception as e:
                    logger.error(f"Retry doc generation failed ({dtype}): {e}")

    random.shuffle(new_docs)
    gold = list(dict.fromkeys(gold_answers))
    wrong = list(set(wrong_answers))

    return {
        "question": task_dict["question"],
        "documents": new_docs,
        "disambig_entity": task_dict["disambig_entity"],
        "gold_answers": gold,
        "wrong_answers": wrong,
        "gold_answers_full": gold,
        "wrong_answers_full": wrong,
    }


def _process_one(
    task_name: str,
    fact: Fact,
    fact_dir: Path,
    distractor_facts_with_dirs: List[Tuple[Fact, Path]],
    company_info: Dict,
    persona: Dict,
    generator: ConflictDatasetGenerator,
    output_dir: str,
    nm: int,
    nt: int,
    nn: int,
    do_verify: bool,
    model: str,
    max_retries: int = 3,
) -> Tuple[Optional[Dict], bool, List[Dict]]:
    """
    Generate + optionally verify one conflict task, with retry on verification failure.

    On each retry, only the failing document types are regenerated; passing docs are kept.
    Returns (task_dict, passed, verify_results).
    """
    try:
        task_dict = generate_task(
            generator=generator,
            question=fact.question,
            supporting_fact=fact,
            supporting_fact_dir=fact_dir,
            distractor_facts_with_dirs=distractor_facts_with_dirs,
            company_info=company_info,
            persona=persona,
            output_dir=output_dir,
            num_misinfo=nm,
            num_temporal=nt,
            num_noise=nn,
            max_workers=2,
        )
    except Exception as e:
        logger.error(f"[{task_name}/{fact.fact_id}] generation failed: {e}")
        return None, False, []

    if not do_verify:
        return task_dict, True, []

    for attempt in range(max_retries + 1):
        passed, verify_results = verify_task(task_dict, model=model, max_workers=2)
        if passed:
            return task_dict, True, verify_results
        if attempt < max_retries:
            failed = [r["doc_type"] for r in verify_results if r["verdict"] == "FAIL"]
            logger.info(
                f"[{task_name}/{fact.fact_id}] attempt {attempt+1} failed "
                f"({failed}), retrying..."
            )
            task_dict = _regenerate_failed_docs(
                task_dict, verify_results, generator,
                fact, fact_dir, distractor_facts_with_dirs,
                company_info, persona, output_dir, nn,
            )

    return task_dict, False, verify_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate conflict dataset from new DRBench task directories"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to data/tasks/ directory (default: auto-discover)",
    )
    parser.add_argument("--output-dir", default="output/conflict_dataset_new")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model for generation and verification "
             f"(default: {DEFAULT_MODEL}). Use 'anthropic/...' for OpenRouter.",
    )
    parser.add_argument("--num-misinfo-per-answer", type=int, default=1)
    parser.add_argument("--num-temporal-per-answer", type=int, default=1)
    parser.add_argument("--num-noise", type=int, default=2)
    parser.add_argument(
        "--random-counts", action="store_true", default=False,
        help="Randomise misinfo/temporal/noise counts",
    )
    parser.add_argument(
        "--no-verify", action="store_true", default=False,
        help="Skip LLM-as-judge verification step",
    )
    parser.add_argument(
        "--save-rejected", action="store_true", default=False,
        help="Write rejected tasks to <output-dir>/rejected.jsonl for inspection",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--max-workers", type=int, default=20,
        help="Number of facts to process in parallel (default: 20)",
    )
    parser.add_argument("--output-format", choices=["jsonl", "json"], default="jsonl")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir / "source" / "tasks"

    task_dirs = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("DR")
    )
    logger.info(f"Found {len(task_dirs)} task directories in {data_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = ConflictDatasetGenerator(model=args.model)

    # ------------------------------------------------------------------
    # Phase 1: load all work items (I/O-only, fast)
    # ------------------------------------------------------------------
    work_items = []
    for task_dir in task_dirs:
        try:
            _, supporting, distractor, company_info, persona = (
                load_facts_from_task_dir(task_dir)
            )
        except Exception as e:
            logger.error(f"Failed to load {task_dir.name}: {e}")
            continue
        if not supporting:
            continue
        for fact, fact_dir in supporting:
            if args.random_counts:
                counts = generate_random_document_counts_multi(num_answers=1)
                pa = counts["per_answer"][0]
                nm, nt, nn = pa["num_misinfo"], pa["num_temporal"], counts["num_noise"]
            else:
                nm = args.num_misinfo_per_answer
                nt = args.num_temporal_per_answer
                nn = args.num_noise
            work_items.append((task_dir.name, fact, fact_dir, distractor,
                                company_info, persona, nm, nt, nn))

    total_facts = len(work_items)
    if args.max_examples:
        work_items = work_items[:args.max_examples]
    logger.info(f"Total supporting facts: {total_facts} "
                f"({'all' if not args.max_examples else args.max_examples} to process)")

    # ------------------------------------------------------------------
    # Phase 2: process all facts in parallel
    # ------------------------------------------------------------------
    accepted: List[Dict] = []
    rejected: List[Dict] = []
    verify_stats: Dict = {"total": 0, "passed": 0, "failed": 0, "by_type": {}}

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                _process_one,
                task_name, fact, fact_dir, distractor, company_info, persona,
                generator, str(output_dir), nm, nt, nn,
                not args.no_verify, args.model,
            ): (task_name, fact)
            for task_name, fact, fact_dir, distractor, company_info, persona, nm, nt, nn
            in work_items
        }

        pbar = tqdm(total=len(futures), desc="Generating facts", unit="fact")
        for future in as_completed(futures):
            task_name, fact = futures[future]
            try:
                task_dict, passed, verify_results = future.result()
            except Exception as e:
                logger.error(f"[{task_name}/{fact.fact_id}] unexpected error: {e}")
                pbar.update(1)
                continue

            if task_dict is None:
                pbar.update(1)
                continue

            if not args.no_verify:
                verify_stats["total"] += 1
                for r in verify_results:
                    dt = r["doc_type"]
                    verify_stats["by_type"].setdefault(dt, {"total": 0, "failed": 0})
                    verify_stats["by_type"][dt]["total"] += 1
                    if r["verdict"] == "FAIL":
                        verify_stats["by_type"][dt]["failed"] += 1

                if passed:
                    verify_stats["passed"] += 1
                    accepted.append(task_dict)
                    pbar.set_postfix({"accepted": len(accepted), "rejected": len(rejected)})
                else:
                    verify_stats["failed"] += 1
                    failures = "; ".join(
                        f"{r['doc_type']}: {r.get('reason','')}"
                        for r in verify_results if r["verdict"] == "FAIL"
                    )
                    tqdm.write(f"  ✗ [{task_name}/{fact.fact_id}] {failures}")
                    if args.save_rejected:
                        task_dict["_verification"] = verify_results
                        rejected.append(task_dict)
            else:
                accepted.append(task_dict)
                pbar.set_postfix({"accepted": len(accepted)})

            pbar.update(1)

        pbar.close()

    # ------------------------------------------------------------------
    # Phase 3: write output
    # ------------------------------------------------------------------
    if not accepted:
        logger.error("No tasks accepted")
        return

    if args.output_format == "jsonl":
        output_file = output_dir / "conflict_dataset.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for task in accepted:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
    else:
        output_file = output_dir / "conflict_dataset.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(accepted, f, indent=2, ensure_ascii=False)

    if args.save_rejected and rejected:
        rej_file = output_dir / "rejected.jsonl"
        with open(rej_file, "w", encoding="utf-8") as f:
            for task in rejected:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
        logger.info(f"Rejected tasks → {rej_file}")

    total_docs = sum(len(t["documents"]) for t in accepted)
    logger.info(f"\n{'='*60}")
    logger.info(f"Done! {len(accepted)} tasks → {output_file}")
    logger.info(f"{'='*60}")

    print(f"\nDataset Statistics:")
    print(f"  Accepted tasks:  {len(accepted)}")
    print(f"  Total documents: {total_docs}")
    print(f"  Avg docs/task:   {total_docs / len(accepted):.1f}")
    print(f"  By type:")
    for dtype in ["correct", "misinfo", "temporal", "noise"]:
        n = sum(
            sum(1 for d in t["documents"] if d["type"] == dtype) for t in accepted
        )
        pct = 100 * n / total_docs if total_docs else 0
        print(f"    {dtype}: {n} ({pct:.1f}%)")

    if not args.no_verify and verify_stats["total"] > 0:
        n = verify_stats["total"]
        acc_pct = 100 * verify_stats["passed"] / n
        print(f"\nVerification (LLM-as-judge):")
        print(f"  Acceptance rate: {verify_stats['passed']}/{n} ({acc_pct:.1f}%)")
        print(f"  Rejection rate:  {verify_stats['failed']}/{n} ({100-acc_pct:.1f}%)")
        print(f"  Failure rate by document type:")
        for dt, s in sorted(verify_stats["by_type"].items()):
            fail_pct = 100 * s["failed"] / s["total"] if s["total"] else 0
            print(f"    {dt}: {s['failed']}/{s['total']} failed ({fail_pct:.1f}%)")


if __name__ == "__main__":
    main()
