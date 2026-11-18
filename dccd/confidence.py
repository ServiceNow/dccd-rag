"""Dual-confidence signals for DCCD.

* **Document-level confidence** ``q_i`` — whether document ``d_i`` appears
  sufficient to answer the question. Computed once per query from a batched
  yes/no support probe (paper Eq. 1 / App. B.1).
* **Token-level confidence** ``c_{i,t}`` — whether the document-conditioned
  next-token distribution is sharp. Computed every decoding step from the
  top-``k`` logits via a lightweight Dirichlet expected-entropy estimate
  (paper Eqs. 2-5 / App. B.2).
"""
from __future__ import annotations

import warnings
from typing import List

import torch
import torch.nn.functional as F

from .models import apply_chat_template
from .prompts import (NO_VERBALIZERS, SUPPORT_PROBE_PROMPT, YES_VERBALIZERS,
                      format_document)


def _single_token_ids(tokenizer, surface_forms) -> List[int]:
    ids = set()
    for s in surface_forms:
        toks = tokenizer(s, add_special_tokens=False)["input_ids"]
        if len(toks) == 1:
            ids.add(toks[0])
    return sorted(ids)


def compute_q_scores(model, tokenizer, question: str, documents, device,
                     no_meta: bool = False, enable_thinking: bool = False,
                     no_chat_template: bool = False) -> List[float]:
    """Document-level confidence ``q_i = sigmoid(L_yes - L_no)`` for each doc.

    ``L_yes`` / ``L_no`` are ``logsumexp`` over the single-token yes / no
    verbalizer ids at the final probe position. All documents for a query are
    scored in one left-padded batched forward pass. ``q`` is computed as
    ``1 - sigmoid(L_no - L_yes)``, exactly as in the paper runner, so the scores
    match it bit-for-bit (the two forms differ in float32 when ``q`` saturates).

    If the tokenizer has no single-token yes or no verbalizer, every document
    gets ``q = 0.5`` (no document-level signal) and a warning is emitted.
    """
    if not documents:
        return []

    yes_ids = _single_token_ids(tokenizer, YES_VERBALIZERS)
    no_ids = _single_token_ids(tokenizer, NO_VERBALIZERS)
    if not yes_ids or not no_ids:
        warnings.warn(
            "DCCD support probe: the tokenizer has no single-token yes/no verbalizer "
            f"(yes={YES_VERBALIZERS}, no={NO_VERBALIZERS}); document-level confidence "
            "falls back to q=0.5 for every document.", UserWarning, stacklevel=2)
        return [0.5] * len(documents)

    probes = []
    for i, d in enumerate(documents):
        doc_str = format_document(i, getattr(d, "title", None), d.text, no_meta=no_meta)
        raw = SUPPORT_PROBE_PROMPT.format(document=doc_str, question=question)
        probes.append(apply_chat_template(tokenizer, raw, enable_thinking, no_chat_template))

    orig_pad = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(probes, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    use_cache=False)
        last = out.logits[:, -1, :].float()
    tokenizer.padding_side = orig_pad

    yes_t = torch.tensor(yes_ids, device=last.device, dtype=torch.long)
    no_t = torch.tensor(no_ids, device=last.device, dtype=torch.long)
    logit_yes = torch.logsumexp(last[:, yes_t], dim=-1)
    logit_no = torch.logsumexp(last[:, no_t], dim=-1)
    c_no = torch.sigmoid(logit_no - logit_yes)
    return [1.0 - float(c) for c in c_no.tolist()]


def dirichlet_expected_entropy(raw_logits_1d: torch.Tensor, top_k: int = 10) -> float:
    """Top-``k`` Dirichlet expected entropy ``H_k`` (paper Eq. 4).

    ``alpha_v = softplus(z_v) + 1`` over the top-``k`` raw logits;
    ``S = sum alpha_v``; ``H_k = sum (alpha_v/S) * [psi(S+1) - psi(alpha_v + 1)]``,
    with ``0 <= H_k <= log k``. Not rescaled by ``log k``.
    """
    k = min(top_k, raw_logits_1d.shape[-1])
    top, _ = torch.topk(raw_logits_1d, k=k, dim=-1)
    top = top.float()
    alpha = F.softplus(top) + 1.0
    S = alpha.sum()
    pbar = alpha / S
    Htilde = (pbar * (torch.digamma(S + 1.0) - torch.digamma(alpha + 1.0))).sum()
    return float(Htilde.item())


def token_confidence(raw_logits_1d: torch.Tensor, top_k: int = 10) -> float:
    """Token-level confidence ``c = 1 - H_k`` (paper Eq. 5), the quantity DCCD
    adds to ``q`` in its dual-confidence score. It lies in ``[1 - log k, 1]``."""
    return 1.0 - dirichlet_expected_entropy(raw_logits_1d, top_k=top_k)


def topk_entropy(logits_1d: torch.Tensor, top_k: int = 10) -> float:
    """Entropy of the top-``k`` probabilities under the full softmax (DVD's
    stream-selection score). Lower = sharper / more confident."""
    log_probs = F.log_softmax(logits_1d, dim=-1)
    probs = log_probs.exp()
    tp, ti = torch.topk(probs, min(top_k, probs.shape[-1]))
    tlp = log_probs.gather(-1, ti)
    return float((-(tp * tlp.clamp(min=-100)).sum()).item())
