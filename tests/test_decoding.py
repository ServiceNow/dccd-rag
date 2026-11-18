"""Decoding: combiner formulas (pinned on crafted tensors), the per-step trace,
the canonical zero-shot path, and truncation warnings."""
import os
import random
import subprocess
import sys

import pytest
import torch

from dccd.decoding import GenConfig, _combine_baseline, _combine_dccd, generate
from dccd.eval import first_sentence
from dccd.prompts import ZEROSHOT_PROMPT
from helpers import make_qa_example

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = 8
FLAT = torch.zeros(V)                                      # highest top-k entropy
SHARP = torch.tensor([40.0, 0, 0, 0, 0, 0, 0, 0])          # lowest
MEDIUM = torch.tensor([8.0, 6, 0, 0, 0, 0, 0, 0])


def _streams():
    """Processed log-prob rows: [closed, full, doc0, doc1, doc2]."""
    g = torch.Generator().manual_seed(0)
    return torch.log_softmax(torch.randn(5, V, generator=g), dim=-1)


def _raw(doc_rows):
    return torch.stack([FLAT, FLAT] + list(doc_rows))


def _run(mode, q, doc_rows=(FLAT, FLAT, FLAT), logits=None, step=0, seed=0):
    logits = _streams() if logits is None else logits
    cfg = GenConfig(mode=mode, random_seed=seed)
    out = _combine_dccd(cfg, logits, _raw(doc_rows), torch.tensor(q), 3, step=step)
    return logits, out


def _direction_pair(logits, out):
    """The (pos, neg) doc pair whose contrast direction produced ``out``."""
    d = out - logits[1]
    best = None
    for p in range(3):
        for n in range(3):
            if p == n:
                continue
            u = logits[2 + p] - logits[2 + n]
            cos = float(torch.dot(d, u) / (d.norm() * u.norm()))
            if best is None or cos > best[0]:
                best = (cos, (p, n))
    assert best[0] > 1 - 1e-5
    return best[1]


# --- DCCD combiner -----------------------------------------------------------

def test_dccd_eps_is_dual_confidence_margin():
    logits, out = _run("dccd", [0.9, 0.5, 0.1])
    assert torch.allclose(out, logits[1] + 0.8 * (logits[2] - logits[4]), atol=1e-6)


def test_dccd_selects_argmax_and_argmin_of_q_minus_au():
    # Equal q: the sharpest stream (lowest AU) is positive, the flattest negative.
    logits, out = _run("dccd", [0.5, 0.5, 0.5], doc_rows=(FLAT, SHARP, MEDIUM))
    assert _direction_pair(logits, out) == (1, 0)


def test_dccd_eps_clipped_to_one():
    logits, out = _run("dccd", [0.995, 0.5, 0.005], doc_rows=(SHARP, MEDIUM, FLAT))
    assert torch.allclose(out, logits[1] + 1.0 * (logits[2] - logits[4]), atol=1e-6)


def test_fixed_gate_eps_one():
    logits, out = _run("fixed_gate", [0.6, 0.5, 0.4])
    assert torch.allclose(out, logits[1] + 1.0 * (logits[2] - logits[4]), atol=1e-6)


def test_token_only_ignores_q():
    logits, out = _run("token_only", [0.9, 0.5, 0.1], doc_rows=(FLAT, MEDIUM, SHARP))
    assert _direction_pair(logits, out) == (2, 0)


def test_doc_only_ignores_au():
    logits, out = _run("doc_only", [0.2, 0.5, 0.8], doc_rows=(SHARP, MEDIUM, FLAT))
    assert torch.allclose(out, logits[1] + 0.6 * (logits[4] - logits[2]), atol=1e-6)


def test_random_uses_seed_plus_step():
    p, n = random.Random(3 + 5).sample(range(3), 2)
    q = [0.5, 0.5, 0.5]
    q[p], q[n] = 0.9, 0.1
    logits, out = _run("random", q, step=5, seed=3)
    assert torch.allclose(out, logits[1] + 0.8 * (logits[2 + p] - logits[2 + n]), atol=1e-6)


def test_nonfinite_falls_back_to_full():
    logits = _streams()
    logits[2, 3] = float("-inf")                       # token 3 outside doc0's nucleus
    _, out = _run("dccd", [0.9, 0.5, 0.1], logits=logits)
    assert out[3] == logits[1, 3]
    keep = [i for i in range(V) if i != 3]
    assert torch.allclose(out[keep], (logits[1] + 0.8 * (logits[2] - logits[4]))[keep], atol=1e-6)


def test_fewer_than_two_docs_returns_full():
    logits = _streams()[:3]
    out = _combine_dccd(GenConfig(mode="dccd"), logits, logits.clone(),
                        torch.tensor([0.7]), 1, step=0)
    assert torch.equal(out, logits[1])


# --- baselines ---------------------------------------------------------------

def test_cad_formula_and_mask():
    logits = _streams()
    logits[0, 2] = -200.0                              # outside the no-context nucleus
    out = _combine_baseline("cad", logits, 5)
    expected = 1.2 * logits[1] - 0.2 * logits[0]
    expected[2] = logits[1, 2]
    assert torch.allclose(out, expected, atol=1e-6)


def test_cocoa_output_is_log_normalized():
    out = _combine_baseline("cocoa", _streams(), 5)
    assert abs(float(torch.logsumexp(out, dim=-1))) < 1e-5


# --- generate(): trace, zero-shot, truncation --------------------------------

def test_trace_does_not_change_prediction(tiny_model):
    model, tok = tiny_model
    ex = make_qa_example(5)
    base = dict(mode="dccd", doc_num=5, max_new_tokens=10, stop_first_sentence=False)
    plain = generate(model, tok, ex, GenConfig(**base))
    traced = generate(model, tok, ex, GenConfig(**base, return_trace=True))
    assert plain["prediction"] == traced["prediction"]
    assert plain["trace"] is None


def test_trace_fields(tiny_model):
    model, tok = tiny_model
    out = generate(model, tok, make_qa_example(5),
                   GenConfig(mode="dccd", doc_num=5, max_new_tokens=10,
                             stop_first_sentence=False, return_trace=True))
    trace = out["trace"]
    assert 1 <= len(trace) <= 10
    assert [t["step"] for t in trace] == list(range(len(trace)))
    # Word-level tokenizer: the joint decode is the space-joined token decodes.
    assert " ".join(t["token"] for t in trace).strip() == out["prediction"]
    for t in trace:
        assert 0 <= t["pos_doc"] < 5 and 0 <= t["neg_doc"] < 5
        assert 0.0 <= t["eps"] <= 1.0


def test_zeroshot_calls_hf_generate_once_unbatched(tiny_model, monkeypatch):
    model, tok = tiny_model
    calls = []
    real = model.generate

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(model, "generate", spy)
    generate(model, tok, make_qa_example(5), GenConfig(mode="zeroshot", max_new_tokens=6))
    assert len(calls) == 1
    assert calls[0]["input_ids"].shape[0] == 1
    assert calls[0]["do_sample"] is False and calls[0]["max_new_tokens"] == 6


def test_zeroshot_matches_hf_generate(tiny_model):
    model, tok = tiny_model
    ex = make_qa_example(5)
    chat = tok.apply_chat_template(
        [{"role": "user", "content": ZEROSHOT_PROMPT.format(question=ex.question)}],
        tokenize=False, add_generation_prompt=True)
    enc = tok(chat, return_tensors="pt")
    ids = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                         max_new_tokens=8, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    manual = tok.decode(ids[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    out = generate(model, tok, ex, GenConfig(mode="zeroshot", max_new_tokens=8,
                                             stop_first_sentence=False))
    assert out["prediction"] == manual


def test_zeroshot_first_sentence_stop_equivalent(tiny_model):
    model, tok = tiny_model
    ex = make_qa_example(5)
    full = generate(model, tok, ex, GenConfig(mode="zeroshot", max_new_tokens=30,
                                              stop_first_sentence=False))["prediction"]
    stop = generate(model, tok, ex, GenConfig(mode="zeroshot", max_new_tokens=30,
                                              stop_first_sentence=True))["prediction"]
    assert first_sentence(stop) == first_sentence(full)


def test_first_sentence_stop_criterion():
    from dccd.decoding import FirstSentenceStop

    class CharTok:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids)

    def ids(text):
        return torch.tensor([[ord(c) for c in "PROMPT" + text]])

    stop = FirstSentenceStop(CharTok(), prompt_len=len("PROMPT"))
    assert not bool(stop(ids("Honda makes the Civic"), None))
    assert bool(stop(ids("Honda makes the Civic. The"), None))
    assert not bool(stop(ids("David J. Ryder"), None))      # initials are not a boundary


def test_truncation_warns(tiny_model):
    model, tok = tiny_model
    with pytest.warns(UserWarning, match="truncat"):
        generate(model, tok, make_qa_example(5),
                 GenConfig(mode="full", doc_num=5, max_length=64, max_new_tokens=8))


@pytest.mark.parametrize("mode", ["DCCD", "Full", "bogus"])
def test_generate_rejects_unknown_mode(tiny_model, mode):
    model, tok = tiny_model
    with pytest.raises(ValueError, match="unknown mode"):
        generate(model, tok, make_qa_example(3), GenConfig(mode=mode, doc_num=3, max_new_tokens=2))


def test_q_modes_derived_from_dccd_modes():
    from dccd.decoding import DCCD_MODES, Q_MODES
    assert set(Q_MODES) == set(DCCD_MODES) - {"token_only"}


def test_dvd_without_documents_returns_full_answer(tiny_model):
    model, tok = tiny_model
    example = make_qa_example(3)
    example.documents = []
    cfg = dict(doc_num=0, max_new_tokens=6, stop_first_sentence=False)
    dvd = generate(model, tok, example, GenConfig(mode="dvd", **cfg))
    full = generate(model, tok, example, GenConfig(mode="full", **cfg))
    assert dvd["prediction"] == full["prediction"]


def test_zeroshot_truncation_warning_names_right_side(tiny_model):
    model, tok = tiny_model
    long_q = make_qa_example(1)
    long_q.question = " ".join(["Which company makes the Civic car model ?"] * 20)
    with pytest.warns(UserWarning, match="right truncation"):
        generate(model, tok, long_q, GenConfig(mode="zeroshot", max_length=24, max_new_tokens=8))


def test_no_clamp_eps():
    assert "clamp_eps" not in GenConfig.__dataclass_fields__
    help_text = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "run_decoding.py"),
                                "--help"], capture_output=True, text=True, check=True).stdout
    assert "--clamp_eps" not in help_text
