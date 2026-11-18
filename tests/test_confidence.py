"""Dual-confidence signals: document-level q (support probe) and token-level
Dirichlet expected entropy."""
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from dccd import confidence as C
from dccd.data import Document

NEG = -1e9


class FakeProbeModel:
    """Returns final-position logits with chosen values on one yes and one no id."""

    def __init__(self, vocab_size, yes_id, no_id, yes_logit, no_logit):
        self.vocab_size, self.yes_id, self.no_id = vocab_size, yes_id, no_id
        self.yes_logit, self.no_logit = yes_logit, no_logit

    def __call__(self, input_ids, attention_mask=None, use_cache=False):
        b, l = input_ids.shape
        logits = torch.full((b, l, self.vocab_size), NEG)
        logits[:, -1, self.yes_id] = self.yes_logit
        logits[:, -1, self.no_id] = self.no_logit
        return SimpleNamespace(logits=logits)


@pytest.fixture(scope="module")
def tok(tiny_ckpt):
    t = AutoTokenizer.from_pretrained(tiny_ckpt)
    t.padding_side = "left"
    return t


def _q(tok, yes_logit, no_logit, n_docs=2):
    yes_id = tok("yes", add_special_tokens=False)["input_ids"][0]
    no_id = tok("no", add_special_tokens=False)["input_ids"][0]
    model = FakeProbeModel(len(tok), yes_id, no_id, yes_logit, no_logit)
    docs = [Document(text="the answer is Honda .", title=f"Doc {i + 1}") for i in range(n_docs)]
    return C.compute_q_scores(model, tok, "Which company makes the Civic ?", docs,
                              torch.device("cpu"))


def test_q_matches_runner_formula_when_saturated(tok):
    q = _q(tok, yes_logit=20.0, no_logit=0.0)
    expected = 1.0 - float(torch.sigmoid(torch.tensor(-20.0)))
    assert q == [expected, expected]
    assert q[0] != 1.0


def test_q_half_when_equal(tok):
    assert _q(tok, yes_logit=3.0, no_logit=3.0) == [0.5, 0.5]


def test_warns_and_returns_half_without_single_token_verbalizers():
    class SplittingTokenizer:
        def __call__(self, text, add_special_tokens=False):
            ids = [7, 8] if text.strip().lower() == "yes" else [9]
            return {"input_ids": ids}

    docs = [Document(text="a"), Document(text="b"), Document(text="c")]
    with pytest.warns(UserWarning, match="verbalizer"):
        q = C.compute_q_scores(None, SplittingTokenizer(), "q?", docs, torch.device("cpu"))
    assert q == [0.5, 0.5, 0.5]


def test_dirichlet_entropy_closed_form_vs_monte_carlo():
    torch.manual_seed(0)
    logits = torch.tensor([3.0, 2.5, 1.0, 0.5, 0.2, -0.3, -1.0, -2.0, -4.0, -6.0, -9.0])
    k = 6
    alpha = torch.nn.functional.softplus(torch.topk(logits, k).values) + 1.0
    samples = torch.distributions.Dirichlet(alpha).sample((200_000,)).clamp_min(1e-12)
    mc = float((-(samples * samples.log()).sum(-1)).mean())
    assert abs(C.dirichlet_expected_entropy(logits, top_k=k) - mc) < 1e-2


def test_token_confidence_is_one_minus_unnormalised_entropy():
    """c = 1 - H_k (paper Eq. 5): the same quantity DCCD adds to q."""
    import math
    logits = torch.tensor([3.0, 2.5, 1.0, 0.5, 0.2, -0.3, -1.0, -2.0, -4.0, -6.0, -9.0])
    hk = C.dirichlet_expected_entropy(logits, top_k=10)
    assert 0.0 <= hk <= math.log(10)
    assert C.token_confidence(logits, top_k=10) == pytest.approx(1.0 - hk)


def test_token_confidence_monotonic():
    """Sharper next-token distribution -> higher confidence c."""
    uniform = torch.zeros(100)
    mild = torch.full((100,), -1e4); mild[3] = 10.0
    sharp = torch.full((100,), -1e4); sharp[3] = 100.0
    c_uni = C.token_confidence(uniform, top_k=10)
    c_mild = C.token_confidence(mild, top_k=10)
    c_sharp = C.token_confidence(sharp, top_k=10)
    assert c_uni <= c_mild <= c_sharp <= 1.0
    assert c_sharp - c_uni > 0.3


def test_topk_entropy_ordering():
    uniform = torch.zeros(50)
    peaked = torch.full((50,), -1e4); peaked[0] = 20.0
    assert C.topk_entropy(uniform, 10) > C.topk_entropy(peaked, 10)
