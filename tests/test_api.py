"""High-level ``DCCD`` API."""
import json
import os

import pytest

from dccd.data import Document, Example
from dccd.decoding import GenConfig, generate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTION = "Which company makes the Civic car model ?"
DOCS = ["Honda is the company that makes the Civic car model .",
        "Toyota makes cars , according to the report .",
        "The chat said Honda makes the Civic ."]


@pytest.fixture(scope="module")
def api(tiny_model):
    from dccd import DCCD
    model, tokenizer = tiny_model
    return DCCD(model, tokenizer, max_new_tokens=8, stop_first_sentence=False)


def test_api_accepts_strings_and_dicts(api):
    as_strings = api.generate(QUESTION, DOCS)
    as_dicts = api.generate(QUESTION, [{"title": f"Doc {i + 1}", "text": t}
                                       for i, t in enumerate(DOCS)])
    assert isinstance(as_strings, str)
    assert as_strings == as_dicts


def test_api_return_trace_shape(api):
    out = api.generate(QUESTION, DOCS, return_trace=True)
    assert set(out) == {"prediction", "q_scores", "trace"}
    assert len(out["q_scores"]) == len(DOCS)
    assert isinstance(out["trace"], list) and out["trace"]


def test_api_matches_low_level_generate(api, tiny_model):
    model, tokenizer = tiny_model
    example = Example(qid="0", question=QUESTION, answers=[],
                      documents=[Document(text=t, title=f"Doc {i + 1}") for i, t in enumerate(DOCS)])
    low = generate(model, tokenizer, example,
                   GenConfig(mode="dccd", doc_num=len(DOCS), max_new_tokens=8,
                             stop_first_sentence=False))
    assert api.generate(QUESTION, DOCS) == low["prediction"]


def test_api_zero_and_one_document(api):
    assert isinstance(api.generate(QUESTION, []), str)
    assert isinstance(api.generate(QUESTION, DOCS[:1]), str)


def test_api_rejects_unknown_mode(api):
    with pytest.raises(ValueError, match="dccd"):
        api.generate(QUESTION, DOCS, mode="DCCD")


def test_api_constructor_mode_default(tiny_model):
    from dccd import DCCD
    model, tokenizer = tiny_model
    rag = DCCD(model, tokenizer, mode="full", return_trace=True, max_new_tokens=8,
               stop_first_sentence=False)
    out = rag.generate(QUESTION, DOCS)
    assert out["prediction"] == DCCD(model, tokenizer, max_new_tokens=8,
                                     stop_first_sentence=False).generate(QUESTION, DOCS, mode="full")
    assert out["q_scores"] is None
    assert isinstance(rag.generate(QUESTION, DOCS, mode="dccd", return_trace=False), str)


def test_api_rejects_bad_constructor_options(tiny_model):
    from dccd import DCCD
    model, tokenizer = tiny_model
    with pytest.raises(ValueError):
        DCCD(model, tokenizer, mode="DCCD")
    with pytest.raises(TypeError, match="max_new_token"):
        DCCD(model, tokenizer, max_new_token=8)


def test_api_untitled_document_gets_default_title(api):
    untitled = [Document(text=t) for t in DOCS]
    assert api.generate(QUESTION, untitled) == api.generate(QUESTION, DOCS)


def test_quickstart_example_exists():
    with open(os.path.join(REPO, "data", "DRQA", "drqa_at10.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert rows[194]["question"] == ("How did Jason Wong respond to a simulated phishing "
                                     "incident in March 2025?")
