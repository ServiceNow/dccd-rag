"""Evaluation: str-em, first-sentence truncation, and the DRQA LLM judge."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dccd import eval as E

EVALUATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "evaluate.py")
RECORDS = {"0": {"question": "Who makes the Civic?", "answers": ["Honda"],
                 "prediction": "Honda is the maker. Toyota is another."}}


class FakeCompletions:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.prompts = reply, exc, []

    async def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        if self.exc is not None:
            raise self.exc
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def fake_client(reply=None, exc=None):
    completions = FakeCompletions(reply, exc)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def _no_sleep(_seconds):
        return None
    monkeypatch.setattr(E, "_SLEEP", _no_sleep)


def _cached(path):
    """Cached (qid, verdict) pairs."""
    try:
        return [(o["qid"], o["correct"]) for o in map(json.loads, open(path))]
    except FileNotFoundError:
        return []


# --- LLM judge ---------------------------------------------------------------

def test_judge_grades_first_sentence(tmp_path):
    client, comp = fake_client('{"correct": true, "reason": "ok"}')
    E.judge_drqa(RECORDS, client=client, cache_path=str(tmp_path / "c.jsonl"))
    assert "Model answer:\nHonda is the maker." in comp.prompts[0]
    assert "Toyota" not in comp.prompts[0]


def test_judge_full_flag_sends_raw(tmp_path):
    client, comp = fake_client('{"correct": true, "reason": "ok"}')
    E.judge_drqa(RECORDS, client=client, cache_path=str(tmp_path / "c.jsonl"),
                 use_first_sentence=False)
    assert "Model answer:\nHonda is the maker. Toyota is another." in comp.prompts[0]


def test_api_errors_not_cached_and_retried(tmp_path):
    cache = str(tmp_path / "c.jsonl")
    bad, bad_comp = fake_client(exc=RuntimeError("upstream 503"))
    r = E.judge_drqa(RECORDS, client=bad, cache_path=cache)
    assert len(bad_comp.prompts) == 3                     # three attempts
    assert r["errors"] == 1 and r["complete"] is False and r["accuracy"] == 0.0
    assert _cached(cache) == []
    good, _ = fake_client('{"correct": true, "reason": "ok"}')
    r = E.judge_drqa(RECORDS, client=good, cache_path=cache)
    assert r["errors"] == 0 and r["complete"] is True and r["accuracy"] == 1.0
    assert _cached(cache) == [("0", True)]


def test_unparseable_counts_incorrect_and_is_cached(tmp_path):
    cache = str(tmp_path / "c.jsonl")
    client, _ = fake_client("I think it is right")
    r = E.judge_drqa(RECORDS, client=client, cache_path=cache)
    assert r["accuracy"] == 0.0 and r["errors"] == 0
    assert _cached(cache) == [("0", False)]
    later, later_comp = fake_client('{"correct": true}')
    assert E.judge_drqa(RECORDS, client=later, cache_path=cache)["accuracy"] == 0.0
    assert later_comp.prompts == []                       # served from the cache


def test_cache_regrades_changed_prediction(tmp_path):
    cache = str(tmp_path / "c.jsonl")
    first, _ = fake_client('{"correct": true}')
    assert E.judge_drqa(RECORDS, client=first, cache_path=cache)["accuracy"] == 1.0
    changed = {"0": {**RECORDS["0"], "prediction": "Toyota makes it."}}
    later, later_comp = fake_client('{"correct": false}')
    r = E.judge_drqa(changed, client=later, cache_path=cache)
    assert len(later_comp.prompts) == 1 and r["accuracy"] == 0.0
    same, same_comp = fake_client('{"correct": true}')          # unchanged: cache hit
    assert E.judge_drqa(changed, client=same, cache_path=cache)["accuracy"] == 0.0
    assert same_comp.prompts == []


def test_cache_regrades_with_other_judge_model(tmp_path):
    cache = str(tmp_path / "c.jsonl")
    first, _ = fake_client('{"correct": true}')
    E.judge_drqa(RECORDS, client=first, cache_path=cache, model="judge-a")
    other, other_comp = fake_client('{"correct": false}')
    r = E.judge_drqa(RECORDS, client=other, cache_path=cache, model="judge-b")
    assert len(other_comp.prompts) == 1 and r["accuracy"] == 0.0
    again, again_comp = fake_client('{"correct": false}')
    assert E.judge_drqa(RECORDS, client=again, cache_path=cache, model="judge-a")["accuracy"] == 1.0
    assert again_comp.prompts == []


def test_rejected_key_fails_fast(tmp_path):
    class AuthenticationError(Exception):
        pass
    records = {str(i): dict(RECORDS["0"]) for i in range(20)}
    client, comp = fake_client(exc=AuthenticationError("invalid key"))
    with pytest.raises(RuntimeError, match="AuthenticationError"):
        E.judge_drqa(records, client=client, cache_path=str(tmp_path / "c.jsonl"), concurrency=1)
    assert len(comp.prompts) == 1                         # no retries, no further requests


def test_parse_string_false():
    assert E._parse_judge('{"correct": "false", "reason": "x"}') is False
    assert E._parse_judge('{"correct": "TRUE"}') is True
    assert E._parse_judge('Sure. {"correct": true, "reason": "same"}') is True
    assert E._parse_judge('{"correct": "maybe"}') is None
    assert E._parse_judge("no json here") is None


def test_missing_key_raises_before_cache_write(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cache = tmp_path / "c.jsonl"
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        E.judge_drqa(RECORDS, cache_path=str(cache))
    assert not cache.exists()


# --- str-em / first sentence -------------------------------------------------

def test_str_em_substring_and_normalization():
    assert E.str_em("The answer is Wilhelm Conrad Röntgen.", ["Wilhelm Conrad Röntgen"])
    assert E.str_em("honda", ["Honda"])
    assert not E.str_em("The maker is Toyota.", ["Honda"])


def test_first_sentence_truncation():
    assert E.first_sentence("Honda is the maker. Toyota is another.") == "Honda is the maker."
    assert "Thomas" in E.first_sentence("It is in St. Thomas which is an island.")


def test_first_sentence_stop_matches_full():
    """A first-sentence generation stop must yield the same string as
    first_sentence(full) on the cases where a naive '.!? + capital' rule diverges."""
    cases = [
        "David J. Ryder is the current Director of the United States Mint.",
        "Built in **Appalachia**. They are isolationist and rarely seen.",
        "What? Yes, it is correct and verified.",
        "1.\nHonda is the maker. Toyota is another.",
        "Honda is the maker. Toyota is another.",
    ]

    def sim_stop(text):
        for end in range(1, len(text) + 1):
            if E.first_sentence_complete(text[:end])[1]:
                return text[:end]
        return text

    for text in cases:
        assert E.first_sentence(sim_stop(text)) == E.first_sentence(text), text


def test_score_str_em_aggregate():
    recs = {"0": {"prediction": "Honda", "answers": ["Honda"]},
            "1": {"prediction": "Toyota", "answers": ["Honda"]}}
    assert E.score_str_em(recs)["accuracy"] == 0.5


# --- CLI ---------------------------------------------------------------------

def test_evaluate_cli_cache_name_follows_mode(tmp_path):
    pred = tmp_path / "p.json"
    pred.write_text(json.dumps(RECORDS))
    entry = {"qid": "0", "correct": True, "judge_model": "anthropic/claude-opus-4-5",
             "prompt_sha256": E._sha256(E._judge_prompt(RECORDS["0"], use_first_sentence=False))}
    (tmp_path / "p.json.judge_full.jsonl").write_text(json.dumps(entry) + "\n")

    def run(*flags):
        return subprocess.run([sys.executable, EVALUATE, "--metric", "judge", "--dry_run",
                               *flags, str(pred)], capture_output=True, text=True,
                              cwd=tmp_path, check=True).stdout

    assert "cached=1" in run("--full")
    assert "cached=0" in run()
