"""DRQA release: loader metadata, counts, the derived @5 bundle, and hygiene."""
import json
import os
import re
import subprocess
import sys

import pytest

from dccd.data import load_dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AT10 = os.path.join(REPO, "data", "DRQA", "drqa_at10.jsonl")


def _derive_at5(out_path: str) -> None:
    subprocess.run([sys.executable, os.path.join(REPO, "drqa", "derive_at5.py"),
                    "--in", AT10, "--out", out_path], check=True, capture_output=True)


@pytest.fixture(scope="module")
def at5_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("at5") / "drqa_at5.jsonl")
    _derive_at5(path)
    return path


def test_drqa_metadata_loaded():
    doc = load_dataset(AT10, "drqa")[0].documents[0]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", doc.document_date)
    assert doc.doc_format in {"md", "email", "chat"}
    assert doc.doc_type in {"correct", "misinfo", "temporal", "noise"}


def test_drqa_counts():
    examples = load_dataset(AT10, "drqa")
    assert len(examples) == 568
    assert sum(ex.doc_size == 10 for ex in examples) == 564
    assert sorted(ex.doc_size for ex in examples if ex.doc_size != 10) == [7, 8, 8, 8]


def test_at5_composition_all_examples(at5_path):
    examples = load_dataset(at5_path, "drqa")
    assert len(examples) == 568
    for ex in examples:
        types = sorted(d.doc_type for d in ex.documents)
        assert types == ["correct", "misinfo", "noise", "noise", "temporal"], ex.qid


def test_at5_deterministic(at5_path, tmp_path):
    again = str(tmp_path / "again.jsonl")
    _derive_at5(again)
    with open(at5_path, "rb") as a, open(again, "rb") as b:
        assert a.read() == b.read()


def test_no_internal_paths():
    with open(AT10, encoding="utf-8") as f:
        for i, line in enumerate(f):
            assert "/home/" not in line and "/Users/" not in line, f"line {i}"


def test_file_paths_resolve():
    source = os.path.join(REPO, "drqa", "source")
    with open(AT10, encoding="utf-8") as f:
        for line in f:
            for doc in json.loads(line)["documents"]:
                if doc.get("file_path"):
                    assert os.path.exists(os.path.join(source, doc["file_path"])), doc["file_path"]
