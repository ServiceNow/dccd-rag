"""Evaluation: normalized string exact-match (str-em) and LLM-as-judge.

* ``str-em`` is used for NQ / TriviaQA / PopQA / RetrievalQA (paper App. C.5).
* ``LLM-as-judge`` is used for DRQA, whose answers are free-form enterprise
  facts (App. C.5). The judge runs through the OpenRouter API.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import string
from typing import Dict, List, Optional

from .prompts import JUDGE_PROMPT

_SLEEP = asyncio.sleep          # retry back-off (patched in tests)

# ---------------------------------------------------------------------------
# Normalized string exact match (SQuAD-style normalization)
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_presence(short_answers: List[str], context: str) -> bool:
    """True iff any normalized gold answer is a substring of the normalized text."""
    n_ctx = normalize_answer(context)
    return any(normalize_answer(a) in n_ctx for a in short_answers if a)


def first_sentence_complete(pred: str):
    """Return ``(sentence, complete)``.

    ``sentence`` is the abbreviation-aware first sentence (paper App. C.3 / C.5):
    the text up to the first newline that follows a content line, or
    the first period preceded by >=3 lowercase letters and followed by a space +
    capital letter, after skipping leading lines with no alphabetic content
    (stray ``!`` / ``1.`` prefixes).

    ``complete`` is True iff a real sentence boundary was consumed (a content
    line was terminated by a newline, or a period boundary matched). It is False
    when ``sentence`` is merely the trailing text with no terminator yet — i.e.
    the first sentence is only final once generation ends. The generation loop
    stops exactly when ``complete`` becomes True, which makes "stop at the first
    sentence" produce the same string as ``first_sentence(full generation)``.
    """
    pred = (pred or "").strip()
    nl_boundary = False
    while "\n" in pred:
        head, rest = pred.split("\n", 1)
        if not re.search(r"[A-Za-z]", head):
            pred = rest.lstrip()
            continue
        pred = head
        nl_boundary = True
        break
    m = re.search(r"(?<=[a-z]{3})\. (?=[A-Z])", pred)
    if m:
        return pred[:m.start() + 1].strip(), True
    return pred.strip(), nl_boundary


def first_sentence(pred: str) -> str:
    """Abbreviation-aware first-sentence truncation (paper App. C.3 / C.5)."""
    return first_sentence_complete(pred)[0]


def str_em(prediction: str, answers: List[str], use_first_sentence: bool = True) -> bool:
    pred = first_sentence(prediction) if use_first_sentence else prediction
    return exact_presence(answers, pred)


def score_str_em(records: Dict[str, dict], use_first_sentence: bool = True) -> dict:
    """Aggregate str-em over a ``{qid: {prediction, answers}}`` result mapping."""
    n = len(records)
    if n == 0:
        return {"n": 0, "accuracy": 0.0}
    correct = sum(
        1 for r in records.values()
        if str_em(r.get("prediction", ""), r.get("answers", []), use_first_sentence)
    )
    return {"n": n, "accuracy": correct / n}


# ---------------------------------------------------------------------------
# LLM-as-judge (DRQA)
# ---------------------------------------------------------------------------

def _parse_judge(text: str) -> Optional[bool]:
    """Extract the boolean ``correct`` field from a judge reply.

    Accepts a JSON boolean or the strings ``"true"``/``"false"`` (any case);
    returns ``None`` when the reply cannot be parsed.
    """
    try:
        m = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
        obj = json.loads(m.group(0)) if m else json.loads(text)
    except Exception:
        return None
    value = obj.get("correct") if isinstance(obj, dict) else None
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


# Errors that retrying cannot fix: stop the whole run instead of re-sending
# every request (matched by class name so ``openai`` stays an optional import).
_FATAL_ERRORS = ("AuthenticationError", "PermissionDeniedError")


def _judge_prompt(rec: dict, use_first_sentence: bool) -> str:
    prediction = rec.get("prediction", "") or ""
    if use_first_sentence:
        prediction = first_sentence(prediction)
    return JUDGE_PROMPT.format(question=rec.get("question", ""),
                               reference_answer=(rec.get("answers") or [""])[0],
                               prediction=prediction)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def judge_drqa(records: Dict[str, dict],
               model: str = "anthropic/claude-opus-4-5",
               api_key: Optional[str] = None,
               base_url: str = "https://openrouter.ai/api/v1",
               cache_path: Optional[str] = None,
               concurrency: int = 16,
               dry_run: bool = False,
               use_first_sentence: bool = True,
               client=None) -> dict:
    """Grade DRQA predictions with an LLM judge (paper App. C.5).

    Returns ``{n, accuracy, verdicts, errors, complete}``. The judge grades
    ``first_sentence(prediction)`` (the paper's single-sentence protocol) unless
    ``use_first_sentence=False``. A reply that cannot be parsed counts as
    incorrect, as in the paper.

    Verdicts are cached to ``cache_path`` (JSONL). A cached verdict is reused
    only when its judge model and the exact judge prompt (question, reference
    and graded text) match, so re-running decoding into the same file, or
    switching ``model``, re-grades instead of reporting stale verdicts. API
    errors are *not* cached: they count as incorrect in this run's
    ``accuracy`` (``complete`` is False) and are retried on the next run. A
    rejected API key stops the run with ``RuntimeError`` (verdicts obtained
    before it are kept). ``client`` may be any object exposing
    ``async chat.completions.create(**kwargs)`` (defaults to an OpenRouter
    ``AsyncOpenAI`` client). With ``dry_run`` no network call is made.
    """
    prompts = {qid: _judge_prompt(rec, use_first_sentence) for qid, rec in records.items()}
    keys = {qid: _sha256(p) for qid, p in prompts.items()}

    cache: Dict[str, bool] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                try:
                    o = json.loads(line)
                    qid = str(o["qid"])
                    if o.get("judge_model") == model and o.get("prompt_sha256") == keys.get(qid):
                        cache[qid] = bool(o["correct"])
                except Exception:
                    pass

    todo = [qid for qid in records if qid not in cache]
    if dry_run:
        return {"n": len(records), "accuracy": None, "pending": len(todo),
                "cached": len(cache), "dry_run": True}

    errors = 0
    if todo:
        if client is None:
            key = api_key or os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY not set (or pass api_key=...).")
            from openai import AsyncOpenAI  # imported lazily
            client = AsyncOpenAI(api_key=key, base_url=base_url)

        fatal: List[BaseException] = []

        async def grade(qid: str, sem: asyncio.Semaphore):
            async with sem:
                for attempt in range(3):
                    if fatal:
                        return qid, None
                    try:
                        resp = await client.chat.completions.create(
                            model=model, temperature=0.0,
                            messages=[{"role": "user", "content": prompts[qid]}],
                            max_tokens=200,
                        )
                        return qid, bool(_parse_judge(resp.choices[0].message.content))
                    except Exception as e:
                        if type(e).__name__ in _FATAL_ERRORS:
                            fatal.append(e)
                            return qid, None
                        if attempt == 2:
                            return qid, None          # API error: not cached
                        await _SLEEP(2 ** attempt)

        async def run() -> int:
            sem = asyncio.Semaphore(concurrency)
            results = await asyncio.gather(*(grade(q, sem) for q in todo))
            graded = [(qid, ok) for qid, ok in results if ok is not None]
            if cache_path and graded:
                with open(cache_path, "a") as f:
                    for qid, ok in graded:
                        f.write(json.dumps({"qid": qid, "correct": ok, "judge_model": model,
                                            "prompt_sha256": keys[qid]}) + "\n")
            cache.update(graded)
            return len(results) - len(graded)

        errors = asyncio.run(run())
        if fatal:
            raise RuntimeError(f"judge API rejected the request ({type(fatal[0]).__name__}: "
                               f"{fatal[0]}); check OPENROUTER_API_KEY.") from fatal[0]

    n = len(records)
    correct = sum(1 for qid in records if cache.get(qid, False))
    return {"n": n, "accuracy": (correct / n if n else 0.0),
            "verdicts": {q: cache.get(q) for q in records},
            "errors": errors, "complete": errors == 0}
