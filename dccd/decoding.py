"""Decoding methods: DCCD, baselines, and ablations.

This is the implementation of the paper's decoding rules (Sec. 4, App. B and C.3)
and mirrors the paper runner (``contrastive_decoding_baselines.py::mode_generate``)
so every column of Table 1 and Table 2 reproduces:

* Every mode except ``zeroshot`` builds the same ``[closed, full, doc_1..doc_n]``
  (n+2) stream batch; streams are left-padded together and advanced with a shared
  KV cache, and the chosen token is replayed on all streams.
* Logits are normalised to log-probs (``z - logsumexp(z)``) and a
  ``LogitsProcessorList`` (top-p 0.95 nucleus + a ``<think>`` ban) is applied to
  all streams before the per-mode combination. Combinations fall back to the
  full-context score wherever an operand is outside its stream's nucleus.
* Greedy decoding = ``argmax`` of the combined scores.
* ``zeroshot`` (question only) uses Hugging Face ``model.generate`` on the single
  closed-book prompt, exactly like the paper's zero-shot runner.

DCCD (``_combine_dccd``)::

    c[j]   = 1 - H_k(z_j)                  token-level confidence (top-k Dirichlet
                                           expected entropy of doc j's raw logits)
    s[j]   = q[j] + c[j]                   q[j]: document-level support probe
    j+     = argmax_j s[j],   j- = argmin_j s[j]
    eps    = min(1, s[j+] - s[j-])
    score  = l_full + eps * (l_{j+} - l_{j-})      l = processed log-probs

The four Table 2 ablations are single-knob variants: ``fixed_gate`` (eps = 1),
``token_only`` (q = 0.5), ``doc_only`` (drop c), ``random`` (random j+/j-).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
import warnings
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import (LogitsProcessorList, StoppingCriteria,
                          StoppingCriteriaList, TopPLogitsWarper)

from .confidence import compute_q_scores, dirichlet_expected_entropy
from .eval import first_sentence_complete
from .models import apply_chat_template, device_of
from .prompts import (CONTEXTUAL_PROMPT, ZEROSHOT_PROMPT, format_document)

# Decoding constants (paper Table 8).
CAD_ALPHA = 0.2
ADACAD_FLOOR = 0.0          # paper adacad runs use --alpha 0.0 (filename _thresh0.0)
DVD_BETA = 0.25
DVD_GAMMA = 0.2
DVD_TOPK = 10
DCCD_TOPK = 10
COCOA = dict(alpha=0.5, gamma=1.0, z=5.0, delta=1e-8, eps=1e-12)
NUCLEUS_TOP_P = 0.95        # nucleus applied to every stream before combining

BASELINE_MODES = ("full", "zeroshot", "cad", "adacad", "cocoa", "dvd")
DCCD_MODES = ("dccd", "fixed_gate", "token_only", "doc_only", "random")
# DCCD-family modes that use the document-level support probe (token_only drops it).
Q_MODES = tuple(m for m in DCCD_MODES if m != "token_only")


@dataclass
class GenConfig:
    mode: str = "dccd"
    doc_num: int = 5
    max_new_tokens: int = 60
    max_length: int = 16384      # large enough that no dataset's prompt is truncated
                                 # (paper used 8000 for NQ/TQA/PopQA/RetrievalQA, 16384 for DRQA;
                                 #  prompts never exceed these, so 16384 reproduces all of them)
    temperature: float = 0.0           # 0 = greedy
    top_p: float = 1.0
    top_k: int = DCCD_TOPK
    no_meta: bool = False
    enable_thinking: bool = False
    no_chat_template: bool = False
    # Stop once the first sentence is complete (eval-consistent: yields the same
    # first_sentence string as generating the full 60 tokens, but faster). Set
    # False to reproduce the raw full-length generations bit-for-bit.
    stop_first_sentence: bool = True
    random_seed: int = 0
    # Record, for DCCD-family modes, the positive/negative document and the
    # gate eps chosen at every step (no effect on the output).
    return_trace: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _think_ban_processor(tokenizer):
    ids = []
    for t in ("<think>", "</think>"):
        e = tokenizer.encode(t, add_special_tokens=False)
        if len(e) == 1:
            ids.append(e[0])
    if not ids:
        return None
    tids = torch.tensor(ids)

    def _proc(input_ids, scores):
        return scores.index_fill_(-1, tids.to(scores.device), float("-inf"))
    return _proc


def _entropy_topk(logp_row: torch.Tensor, k: int) -> float:
    """DVD's per-stream score: top-k entropy on log-prob logits (verbatim)."""
    probas = F.softmax(logp_row[None], dim=-1)
    _, indices = torch.topk(probas, k, largest=True)
    V = logp_row[indices.squeeze(0)]
    return float((-(V.exp() * V.clip(-100, 0)).sum()).item())


def _jsd_from_logp(logp_a: torch.Tensor, logp_b: torch.Tensor) -> float:
    p = F.softmax(logp_a, dim=-1)
    q = F.softmax(logp_b, dim=-1)
    m = (0.5 * (p + q)).clamp_min(1e-9).log()
    ps, qs = p.clamp_min(1e-9), q.clamp_min(1e-9)
    return float((0.5 * ((ps * (ps.log() - m)).sum() + (qs * (qs.log() - m)).sum())).item())


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def _warn_if_truncated(tokenizer, prompts: List[str], max_prompt_len: int,
                       side: str = "left") -> None:
    longest = max(len(ids) for ids in tokenizer(prompts)["input_ids"])
    if longest > max_prompt_len:
        lost = ("the beginning of the prompt (including the instruction)" if side == "left"
                else "the end of the prompt (including the answer cue)")
        warnings.warn(
            f"Prompt truncated from {longest} to {max_prompt_len} tokens; {side} "
            f"truncation drops {lost}. Increase max_length to avoid this.",
            UserWarning, stacklevel=3)


class FirstSentenceStop(StoppingCriteria):
    """Stop ``model.generate`` once the first sentence of the new text is complete
    (same rule as the eval-time :func:`dccd.eval.first_sentence`)."""

    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        done = [first_sentence_complete(
                    self.tokenizer.decode(row[self.prompt_len:], skip_special_tokens=True))[1]
                for row in input_ids]
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


def _generate_zeroshot(model, tokenizer, example, cfg: GenConfig) -> dict:
    """Closed-book answer with Hugging Face ``model.generate``: the procedure of
    the paper's zero-shot runner (one prompt at a time, greedy, chat template
    with thinking off, ``<think>`` blocks stripped from the output)."""
    device = device_of(model)
    prompt = apply_chat_template(tokenizer, ZEROSHOT_PROMPT.format(question=example.question),
                                 cfg.enable_thinking, cfg.no_chat_template)
    max_prompt_len = max(1, cfg.max_length - cfg.max_new_tokens)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=max_prompt_len).to(device)
    prompt_len = enc.input_ids.shape[1]
    if prompt_len >= max_prompt_len:
        _warn_if_truncated(tokenizer, [prompt], max_prompt_len,
                           side=getattr(tokenizer, "truncation_side", "right"))
    kwargs = dict(max_new_tokens=cfg.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    if cfg.temperature > 0:
        kwargs.update(do_sample=True, temperature=cfg.temperature)
    else:
        kwargs.update(do_sample=False, temperature=None, top_p=None)
    if cfg.stop_first_sentence:
        kwargs["stopping_criteria"] = StoppingCriteriaList([FirstSentenceStop(tokenizer, prompt_len)])
    out = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, **kwargs)
    prediction = _strip_think(
        tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip())
    return {
        "qid": example.qid,
        "question": example.question,
        "answers": example.answers,
        "wrong_answers": getattr(example, "wrong_answers", []),
        "prediction": prediction,
        "mode": cfg.mode,
        "doc_num": cfg.doc_num,
        "q_scores": None,
        "trace": None,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, example, cfg: GenConfig) -> dict:
    if cfg.mode not in BASELINE_MODES + DCCD_MODES:
        raise ValueError(f"unknown mode {cfg.mode!r}; choose one of "
                         f"{', '.join(BASELINE_MODES + DCCD_MODES)}")
    if cfg.mode == "zeroshot":
        return _generate_zeroshot(model, tokenizer, example, cfg)
    device = device_of(model)
    docs = example.documents[:cfg.doc_num]
    n_docs = len(docs)

    # --- build the (n+2)-stream batch: [closed, full, doc_1..doc_n] ----------
    formatted = [format_document(i, getattr(d, "title", None), d.text, no_meta=cfg.no_meta)
                 for i, d in enumerate(docs)]
    closed = ZEROSHOT_PROMPT.format(question=example.question)
    full = CONTEXTUAL_PROMPT.format(question=example.question,
                                    search_results="\n".join(formatted))
    per_doc = [CONTEXTUAL_PROMPT.format(question=example.question, search_results=fd)
               for fd in formatted]
    raw_batch = [closed, full] + per_doc
    batch = [apply_chat_template(tokenizer, p, cfg.enable_thinking, cfg.no_chat_template)
             for p in raw_batch]

    # --- document-level confidence q_i (DCCD-family; token_only zeroes it) ----
    if cfg.mode in Q_MODES and n_docs > 0:
        q_scores = compute_q_scores(model, tokenizer, example.question, docs, device,
                                    no_meta=cfg.no_meta, enable_thinking=cfg.enable_thinking,
                                    no_chat_template=cfg.no_chat_template)
    else:
        q_scores = [0.5] * n_docs
    q_tensor = torch.tensor(q_scores, dtype=torch.float32, device=device)

    # --- processors: top-p warper + <think> ban (applied to all streams) -----
    processors = LogitsProcessorList([TopPLogitsWarper(min(NUCLEUS_TOP_P, cfg.top_p))])
    if not cfg.enable_thinking:
        ban = _think_ban_processor(tokenizer)
        if ban is not None:
            processors.append(ban)

    # --- tokenize (left padding + left truncation), matching the runner ------
    prev_trunc = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    max_prompt_len = max(1, cfg.max_length - cfg.max_new_tokens)
    enc = tokenizer(batch, padding="longest", truncation=True,
                    max_length=max_prompt_len, return_tensors="pt").to(device)
    tokenizer.truncation_side = prev_trunc
    if enc.input_ids.shape[1] >= max_prompt_len:
        _warn_if_truncated(tokenizer, batch, max_prompt_len)
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask
    n = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    max_tokens = max(1, min(cfg.max_new_tokens, cfg.max_length - prompt_len))

    past = None
    gen: List[int] = []
    eos_id = tokenizer.eos_token_id
    trace = [] if (cfg.return_trace and cfg.mode in DCCD_MODES) else None

    for _ in range(max_tokens):
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    return_dict=True, use_cache=True, past_key_values=past,
                    logits_to_keep=1)
        past = out.past_key_values
        raw = out.logits[:, -1]                                  # [n, V] pre-softmax
        logp = raw - raw.logsumexp(dim=-1, keepdim=True)         # log-probs
        proc = processors(input_ids, logp.clone())               # warped + think-banned

        if cfg.mode in BASELINE_MODES:
            final = _combine_baseline(cfg.mode, proc, n)
        else:
            # DCCD-family uses the SAME processed log-probs as the baselines for
            # its streams (so <think>/top-p masking is already applied); raw
            # logits feed the token-level AU. step = #tokens so far (random seed).
            step_info = {} if trace is not None else None
            final = _combine_dccd(cfg, proc, raw, q_tensor, n_docs, step=len(gen),
                                  trace_step=step_info)

        tok = int(torch.argmax(final).item()) if cfg.temperature <= 0 else \
            int(torch.multinomial(F.softmax(final / cfg.temperature, -1), 1).item())

        if tok == eos_id:
            break
        gen.append(tok)
        if trace is not None:
            trace.append({"step": len(gen) - 1, "token": tokenizer.decode([tok]),
                          "pos_doc": step_info.get("pos_doc"),
                          "neg_doc": step_info.get("neg_doc"),
                          "eps": step_info.get("eps")})
        if cfg.stop_first_sentence:
            # Stop only once a real sentence boundary has been consumed, using
            # the SAME logic as the eval-time first_sentence(). This guarantees
            # the result equals first_sentence(full generation) — verified to
            # match on 135k cached generations, whereas a naive "\\n or .!? +
            # capital" rule diverges ~14% of the time (initials, markdown before
            # the period, ?/!, leading non-alpha lines).
            if first_sentence_complete(tokenizer.decode(gen))[1]:
                break
        next_tok = torch.tensor([tok], device=device).unsqueeze(-1).tile(n, 1)
        input_ids = next_tok
        attention_mask = torch.cat(
            [attention_mask, torch.ones(n, 1, dtype=torch.long, device=device)], dim=-1)

    prediction = _strip_think(tokenizer.decode(gen, skip_special_tokens=True)).strip()
    return {
        "qid": example.qid,
        "question": example.question,
        "answers": example.answers,
        "wrong_answers": getattr(example, "wrong_answers", []),
        "prediction": prediction,
        "mode": cfg.mode,
        "doc_num": cfg.doc_num,
        "q_scores": q_scores if cfg.mode in Q_MODES else None,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Baseline combiners — verbatim from contrastive_decoding_baselines.py
# ---------------------------------------------------------------------------

def _combine_baseline(mode: str, logits: torch.Tensor, n: int) -> torch.Tensor:
    logits_uncond = logits[0]
    logits_full = logits[1]

    if mode == "full":
        return logits_full
    if mode == "zeroshot":
        return logits_uncond

    if mode == "cad":
        merged = (1 + CAD_ALPHA) * logits_full - CAD_ALPHA * logits_uncond
        return torch.where(logits_uncond > -100, merged, logits_full)

    if mode == "adacad":
        jsd = _jsd_from_logp(logits_full, logits_uncond)
        alpha_t = max(jsd, ADACAD_FLOOR)
        merged = (1 + alpha_t) * logits_full - alpha_t * logits_uncond
        return torch.where(logits_uncond > -100, merged, logits_full)

    if mode == "cocoa":
        return _cocoa(logits_uncond, logits_full)

    if mode == "dvd":
        return _dvd(logits, n)

    raise ValueError(f"unknown baseline mode {mode!r}")


def _dvd(logits: torch.Tensor, n: int) -> torch.Tensor:
    """DVD static-weight (Jin et al. 2024), verbatim from the runner."""
    logits_uncond = logits[0]
    logits_full = logits[1]
    if n < 3:                       # no retrieved document: nothing to contrast
        return logits_full
    cad_entropy = _entropy_topk(logits[0], DVD_TOPK)
    full_entropy = _entropy_topk(logits[1], DVD_TOPK)

    logits_max = logits[2]
    logits_min = logits[n - 1]
    maxent, minent = -0.01, 50.0
    for i in range(2, n):
        ent = _entropy_topk(logits[i], DVD_TOPK)
        if ent > maxent:
            maxent = ent
            logits_min = logits[i]
        if ent < minent:
            minent = ent
            logits_max = logits[i]

    beta_t, gamma_t = DVD_BETA, DVD_GAMMA
    if cad_entropy * 10 < full_entropy:
        merged = logits_full + gamma_t * logits_max - gamma_t * logits_min
        return torch.where(logits_min > -100, merged, logits_full)
    merged = ((1 + beta_t) * logits_full - beta_t * logits_uncond
              + gamma_t * logits_max - gamma_t * logits_min)
    final = torch.where(logits_uncond > -100, merged, logits_full)
    return torch.where(logits_min > -100, final, logits_full)


def _cocoa(logits_uncond: torch.Tensor, logits_full: torch.Tensor) -> torch.Tensor:
    """Paper-faithful CoCoA, verbatim from the runner. Inputs are the processed
    (top-p-masked) log-prob rows; CoCoA re-applies ``log_softmax`` to them."""
    a, g, z, delta, eps = (COCOA["alpha"], COCOA["gamma"], COCOA["z"],
                           COCOA["delta"], COCOA["eps"])
    log_p_prior = F.log_softmax(logits_uncond.float(), dim=-1)
    log_p_ctx = F.log_softmax(logits_full.float(), dim=-1)
    valid = logits_uncond > -100
    log_p_prior = torch.where(valid, log_p_prior, log_p_ctx)
    p_prior = log_p_prior.exp()
    p_ctx = log_p_ctx.exp()

    log_renyi = torch.logsumexp(a * log_p_prior + (1.0 - a) * log_p_ctx, dim=-1)
    d_alpha = (log_renyi / (a - 1.0)).clamp_min(0.0)
    zero = torch.zeros_like(log_p_prior)
    h_prior = -(torch.where(log_p_prior > -1e30, p_prior * log_p_prior, zero)).sum(-1)
    h_ctx = -(torch.where(log_p_ctx > -1e30, p_ctx * log_p_ctx, zero)).sum(-1)
    delta_h = h_prior - h_ctx
    s_t = torch.sigmoid(d_alpha + g * delta_h + delta).clamp(eps, 1.0 - eps)
    top2 = torch.topk(p_ctx, k=2, dim=-1).values
    m_t = (top2[0] - top2[1]).clamp(eps, 1.0)
    conflict = torch.log(s_t) - torch.log1p(-s_t)
    lam = torch.sigmoid(z * torch.log(m_t) + conflict)
    log_q = (1.0 - lam) * log_p_prior + lam * log_p_ctx
    return F.log_softmax(log_q, dim=-1)


# ---------------------------------------------------------------------------
# DCCD combiner — the exact config that produced the paper's Table 1 / Table 2
# (reference runner: dvd_dual_pick + dual_dual_jsd + dual_eps_score_margin +
#  dual_score_criterion='au', λ=μ=1, K=1). Streams use the processed log-probs
# `logits` (top-p + <think> ban, as the baselines); raw logits feed AU.
#
#   AU[j]        = top-k Dirichlet expected entropy H_k of raw logits (NOT ÷log k)
#   score_pos[j] = q[j]     − λ·AU[j]
#   score_neg[j] = (1−q[j]) + μ·AU[j]
#   j⁺ = argmax score_pos ; j⁻ = argmax score_neg            (first-max, like Python max)
#   ε  = clamp(score_pos[j⁺] − score_pos[j⁻], 0, 1)
#   z  = z_full + ε·(z[j⁺] − z[j⁻]) ; final = where(isfinite, z, z_full)
#
# Ablations (Table 2): fixed_gate → ε≡1 ; token_only → q≡0.5 ; doc_only → λ=μ=0 ;
# random → j⁺,j⁻ uniform-random (seed = random_seed + step).
# ---------------------------------------------------------------------------

def _combine_dccd(cfg: GenConfig, logits: torch.Tensor, raw: torch.Tensor,
                  q_tensor: torch.Tensor, n_docs: int, step: int,
                  trace_step: Optional[dict] = None) -> torch.Tensor:
    z_full = logits[1]
    if n_docs < 2:
        return z_full

    au = [dirichlet_expected_entropy(raw[2 + j], top_k=cfg.top_k) for j in range(n_docs)]
    if cfg.mode == "token_only":                 # no_q_scores: drop document confidence
        q = [0.5] * n_docs
    else:
        q = [float(q_tensor[j]) for j in range(n_docs)]
    lam = mu = (0.0 if cfg.mode == "doc_only" else 1.0)   # doc_only: drop token AU

    score_pos = [q[j] - lam * au[j] for j in range(n_docs)]
    score_neg = [(1.0 - q[j]) + mu * au[j] for j in range(n_docs)]

    if cfg.mode == "random":
        j_pos, j_neg = random.Random(cfg.random_seed + step).sample(range(n_docs), 2)
    else:
        j_pos = max(range(n_docs), key=lambda j: score_pos[j])
        j_neg = max(range(n_docs), key=lambda j: score_neg[j])

    if cfg.mode == "fixed_gate":
        eps = 1.0
    else:
        eps = max(0.0, min(1.0, score_pos[j_pos] - score_pos[j_neg]))
    if trace_step is not None:                    # record only; no effect on output
        trace_step.update(pos_doc=j_pos, neg_doc=j_neg, eps=float(eps))

    merged = z_full + eps * (logits[2 + j_pos] - logits[2 + j_neg])
    return torch.where(torch.isfinite(merged), merged, z_full)
