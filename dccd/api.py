"""High-level API: answer a question from retrieved documents with DCCD.

    from dccd import DCCD

    rag = DCCD("Qwen/Qwen3.5-2B")
    answer = rag.generate("Who makes the Civic?", ["Honda makes the Civic.", "..."])

Every call goes through the same :func:`dccd.decoding.generate` used to produce
the paper's results, with the paper's decoding settings by default.
"""
from __future__ import annotations

import dataclasses
from typing import Mapping, Optional, Sequence, Union

from . import ALL_MODES
from .data import Document, Example
from .decoding import GenConfig, generate
from .models import load_model_and_tokenizer

DocumentLike = Union[str, Mapping[str, str], Document]


def _to_document(doc: DocumentLike, index: int) -> Document:
    default_title = f"Doc {index + 1}"
    if isinstance(doc, Document):
        return doc if doc.title else dataclasses.replace(doc, title=default_title)
    if isinstance(doc, str):
        return Document(text=doc, title=default_title)
    if isinstance(doc, Mapping):
        return Document(text=doc["text"], title=doc.get("title") or default_title)
    raise TypeError(f"documents must be str, dict or Document, got {type(doc).__name__}")


def _check_mode(mode: str) -> None:
    if mode not in ALL_MODES:
        raise ValueError(f"unknown mode {mode!r}; choose one of {', '.join(ALL_MODES)}")


class DCCD:
    """Dual-Confidence Contrastive Decoding over any Hugging Face causal LM.

    Args:
        model: a model id / local path, or an already loaded model.
        tokenizer: required when ``model`` is a model object.
        dtype, device_map: used when loading from a model id or path.
        trust_remote_code: allow the checkpoint's own modelling code from the
            Hub (off by default; the paper's models do not need it).
        mode: default decoding mode for :meth:`generate` (``"dccd"``).
        **gen_overrides: default :class:`~dccd.decoding.GenConfig` fields for
            every call (e.g. ``max_new_tokens=60``, ``stop_first_sentence=False``).
    """

    def __init__(self, model, tokenizer=None, *, dtype: str = "bfloat16",
                 device_map: str = "auto", trust_remote_code: bool = False,
                 mode: str = "dccd", **gen_overrides):
        _check_mode(mode)
        fields = {f.name for f in dataclasses.fields(GenConfig)} - {"mode"}
        unknown = sorted(set(gen_overrides) - fields)
        if unknown:
            raise TypeError(f"unknown generation option(s): {', '.join(unknown)}")
        if isinstance(model, str):
            self.model, self.tokenizer = load_model_and_tokenizer(
                model, dtype=dtype, device_map=device_map,
                trust_remote_code=trust_remote_code)
        else:
            if tokenizer is None:
                raise ValueError("pass `tokenizer` together with a model object")
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self.model, self.tokenizer = model, tokenizer
        self.mode = mode
        self.gen_overrides = gen_overrides

    def generate(self, question: str, documents: Sequence[DocumentLike],
                 mode: Optional[str] = None, return_trace: Optional[bool] = None,
                 **gen_overrides):
        """Answer ``question`` from ``documents``.

        ``documents`` may be strings, ``{"title", "text"}`` dicts, or
        :class:`~dccd.data.Document` objects. ``mode`` is any decoding mode in
        :data:`dccd.ALL_MODES` (default: the constructor's ``mode``). Returns
        the answer string, or ``{"prediction", "q_scores", "trace"}`` when
        ``return_trace=True`` (the trace lists, per generated token, the
        positive and negative document (0-based) and the gate eps).
        """
        mode = self.mode if mode is None else mode
        _check_mode(mode)
        options = {**self.gen_overrides, **gen_overrides}
        if return_trace is not None:
            options["return_trace"] = return_trace
        docs = [_to_document(d, i) for i, d in enumerate(documents)]
        options.setdefault("doc_num", len(docs))
        cfg = GenConfig(mode=mode, **options)
        example = Example(qid="0", question=question, answers=[], documents=docs)
        out = generate(self.model, self.tokenizer, example, cfg)
        if cfg.return_trace:
            return {"prediction": out["prediction"], "q_scores": out["q_scores"],
                    "trace": out["trace"]}
        return out["prediction"]
