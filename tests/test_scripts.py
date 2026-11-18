"""Command-line scripts: data preparation and reproduction drivers."""
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sandbox(tmp_path):
    """A throwaway copy of the code (and a link to DRQA) so scripts that write
    into <repo>/data do not touch the real checkout."""
    root = tmp_path / "repo"
    for sub in ("dccd", "scripts"):
        shutil.copytree(os.path.join(REPO, sub), root / sub)
    (root / "drqa").mkdir()
    shutil.copy(os.path.join(REPO, "drqa", "derive_at5.py"), root / "drqa")
    (root / "data" / "DRQA").mkdir(parents=True)
    os.symlink(os.path.join(REPO, "data", "DRQA", "drqa_at10.jsonl"),
               root / "data" / "DRQA" / "drqa_at10.jsonl")
    return root


def test_verify_data_converts_fid_layout(tmp_path):
    # FiD's preprocess.py writes open_domain_data/NQ/test.json and .../TQA/test.json.
    fid = tmp_path / "open_domain_data"
    example = [{"question": "q?", "answers": ["a"], "target": "a",
                "ctxs": [{"id": "1", "title": "t", "text": "x", "score": 1.0}]}]
    for sub in ("NQ", "TQA"):
        (fid / sub).mkdir(parents=True)
        (fid / sub / "test.json").write_text(json.dumps(example))
    root = _sandbox(tmp_path)
    subprocess.run([sys.executable, str(root / "scripts" / "verify_data.py"),
                    "--fid_dir", str(fid)], capture_output=True, text=True)
    for sub in ("NQ", "TQA"):
        converted = json.loads((root / "data" / sub / "test.json").read_text())
        assert converted[0]["ctxs"] == [{"id": "1", "title": "t", "text": "x"}], sub


def _dry_run(script, *args):
    env = {**os.environ, "DRY_RUN": "1"}
    proc = subprocess.run(["/bin/bash", os.path.join(REPO, "scripts", script), *args],
                          capture_output=True, text=True, env=env, cwd=REPO)
    assert proc.returncode == 0, proc.stderr
    return [line for line in proc.stdout.splitlines() if "scripts/run_decoding.py" in line]


def _check_drqa_depths(lines):
    for line in (l for l in lines if "--dataset drqa" in l):
        if "--doc_num 10" in line:
            assert "drqa_at10.jsonl" in line, line
        else:
            assert "--doc_num 5" in line and "drqa_at5.jsonl" in line, line


def test_reproduce_main_dry_run_command_count(tmp_path):
    lines = _dry_run("reproduce_main.sh", "qwen3.5-2b", str(tmp_path))
    assert len(lines) == 5 * (1 + 2 * 6)
    assert all("--model Qwen/Qwen3.5-2B" in line for line in lines)
    assert sum("--mode zeroshot" in line for line in lines) == 5
    _check_drqa_depths(lines)


def test_reproduce_main_rejects_unknown_model(tmp_path):
    env = {**os.environ, "DRY_RUN": "1"}
    proc = subprocess.run(["/bin/bash", os.path.join(REPO, "scripts", "reproduce_main.sh"),
                           "gpt-17", str(tmp_path)], capture_output=True, text=True, env=env, cwd=REPO)
    assert proc.returncode != 0 and "qwen3.5-2b" in proc.stderr


def test_reproduce_ablation_dry_run_command_count(tmp_path):
    lines = _dry_run("reproduce_ablation.sh", str(tmp_path))
    assert len(lines) == 5 * 2 * 5
    assert all("--model Qwen/Qwen3.5-2B" in line for line in lines)
    _check_drqa_depths(lines)


def test_cli_help():
    for script in ("run_decoding.py", "evaluate.py", "verify_data.py", "download_data.py"):
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", script), "--help"],
                       capture_output=True, check=True)


def test_package_metadata():
    import pytest
    tomllib = pytest.importorskip("tomllib")
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as f:
        meta = tomllib.load(f)["project"]
    assert meta["name"] == "dccd-rag"
    assert meta["requires-python"] == ">=3.10"
    assert "Apache" in str(meta["license"])
