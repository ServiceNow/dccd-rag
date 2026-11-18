"""Dataset loading for DCCD experiments.

Two on-disk formats are supported:

* **Standard multi-document QA** (``nq`` / ``tqa`` / ``popqa`` / ``retrievalqa``):
  a JSON array of ``{"question", "answers", "ctxs": [{"title", "text", ...}]}``.
* **DRQA** (``drqa``): RAMDocs-style JSONL, one object per line with
  ``{"question", "documents": [{"text", "type", ...}], "gold_answers",
  "wrong_answers"}``.

Both yield a uniform list of :class:`Example` objects, each holding a list of
:class:`Document` objects. The decoder slices ``example.documents[:doc_num]`` to
realise the @5 / @10 retrieval depths.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Document:
    text: str
    title: Optional[str] = None
    # DRQA metadata. It is kept for analysis and is never shown to the model:
    # prompts contain only ``Document [i](Title: <title>) <text>``.
    doc_type: Optional[str] = None       # correct / misinfo / temporal / noise
    document_date: Optional[str] = None  # ISO date assigned to the document
    doc_format: Optional[str] = None     # md / email / chat


@dataclass
class Example:
    qid: str
    question: str
    answers: List[str]
    documents: List[Document]
    wrong_answers: List[str] = field(default_factory=list)

    @property
    def doc_size(self) -> int:
        return len(self.documents)


# Canonical dataset-type aliases.
QA_TYPES = {"nq", "tqa", "popqa", "retrievalqa", "qa"}
DRQA_TYPES = {"drqa", "ramdocs"}


def load_dataset(path: str, dtype: str) -> List[Example]:
    """Load ``path`` as a list of :class:`Example` for dataset type ``dtype``."""
    dtype = dtype.lower()
    if dtype in QA_TYPES:
        return _load_qa(path)
    if dtype in DRQA_TYPES:
        return _load_drqa(path)
    raise ValueError(f"Unknown dataset type {dtype!r}; expected one of "
                     f"{sorted(QA_TYPES | DRQA_TYPES)}")


def _load_qa(path: str) -> List[Example]:
    with open(path, "r") as f:
        data = json.load(f)
    examples = []
    for i, item in enumerate(data):
        docs = [Document(text=c["text"], title=c.get("title"))
                for c in item.get("ctxs", [])]
        answers = item["answers"]
        if isinstance(answers, str):
            answers = [answers]
        examples.append(Example(
            qid=str(item.get("_id", item.get("id", i))),
            question=item["question"],
            answers=list(answers),
            documents=docs,
        ))
    return examples


def _load_drqa(path: str) -> List[Example]:
    examples = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            docs = [
                Document(text=d["text"],
                         title=d.get("title") or f"Doc {j + 1}",
                         doc_type=d.get("type"),
                         document_date=d.get("document_date"),
                         doc_format=d.get("format"))
                for j, d in enumerate(item.get("documents", []))
            ]
            examples.append(Example(
                qid=str(i),
                question=item["question"],
                answers=list(item.get("gold_answers") or []),
                documents=docs,
                wrong_answers=list(item.get("wrong_answers") or []),
            ))
    return examples


class QADataset:
    """Thin sequence wrapper around :func:`load_dataset`."""

    def __init__(self, path: str, dtype: str):
        self.path = path
        self.dtype = dtype
        self.examples = load_dataset(path, dtype)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i) -> Example:
        return self.examples[i]

    def __iter__(self):
        return iter(self.examples)
