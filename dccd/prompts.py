"""Prompt templates — byte-for-byte identical to the paper's runner
(``contrastive_decoding_baselines.py``) so decoding reproduces the published
results exactly. The trailing spaces after ``Answer:`` are intentional and
must not be removed: they change tokenization and therefore the model output.
"""
from __future__ import annotations

# --- Answer generation (paper App. C.2) -------------------------------------------
# NOTE: trailing space after "Answer:" is significant (matches MD_PROMPT /
# CLOSED_BOOK_PROMPT in the paper runner).

ZEROSHOT_PROMPT = "Question: {question}\nAnswer: "

CONTEXTUAL_PROMPT = (
    "Write a high-quality concise answer for the given question using only the "
    "provided search results (some of which might be irrelevant).\n\n"
    "{search_results}\n\n"
    "Question: {question}\nAnswer: "
)

# --- Document-level support probe for DCCD (paper App. B.1) ----------------------
# Matches `_compute_doc_q_scores` in the paper runner.

SUPPORT_PROBE_PROMPT = (
    "{document}\n\n"
    "Question: {question}\n"
    "Does this document contain enough information to answer the question? (yes/no):\n"
    "Answer: "
)

# Yes / No verbalizer surface forms (single-token variants kept at runtime).
YES_VERBALIZERS = ["yes", "Yes", "YES", " yes", " Yes", " YES"]
NO_VERBALIZERS = ["no", "No", "NO", " no", " No", " NO"]

# --- LLM-as-judge for DRQA (paper App. C.5) --------------------------------------

JUDGE_PROMPT = """You are a strict but fair grader for short-answer question answering. Decide whether the model answer matches the reference answer.

Question:
{question}

Reference answer:
{reference_answer}

Model answer:
{prediction}

Mark the model answer correct if and only if it conveys the meaning of the reference answer.

Rules:
1. Surface form does not matter if the meaning is equivalent.
2. Numerical and date answers must match the reference at the level of detail provided.
3. A refusal or "not specified" answer is incorrect unless the reference answer also says the information is unavailable.
4. Extra context is acceptable only if the answer clearly commits to the correct value.
5. If the answer gives a wrong primary value but mentions the correct value only incidentally, mark it incorrect.

Return exactly one JSON object:
{{"correct": true_or_false, "reason": "<short reason>"}}"""


def format_document(idx: int, title, text: str, no_meta: bool = False) -> str:
    """Render one document. Matches ``Prompt._format_doc`` in the paper runner:
    ``Document [i](Title: <title>) <text>`` (1-based ``i``), or bare text."""
    if no_meta:
        return text
    return f"Document [{idx + 1}](Title: {title}) {text}"


def build_search_results(documents, no_meta: bool = False, start: int = 0) -> str:
    """Join formatted documents for the ``{search_results}`` slot. ``start`` sets
    the 1-based document index offset so per-document streams keep their original
    position label (the paper builds all docs once, then slices by cluster)."""
    return "\n".join(
        format_document(start + i, getattr(d, "title", None), d.text, no_meta=no_meta)
        for i, d in enumerate(documents)
    )
