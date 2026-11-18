#!/usr/bin/env python3
"""Obtain (if requested) and verify the evaluation datasets.

Only DRQA (the paper's benchmark) ships with the repo; the large standard-QA sets
are obtained from their original sources via ``scripts/download_data.py``. This script fetches the auto-downloadable ones,
verifies every dataset's example count and schema, and prints a PASS/SKIP/FAIL
table.

Usage:
  python scripts/verify_data.py                 # verify whatever is present
  python scripts/verify_data.py --download       # also fetch PopQA + RetrievalQA from HF
  python scripts/verify_data.py --download \
        --fid_dir /path/to/FiD/open_domain_data  # also reformat NQ + TriviaQA
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dccd.data import load_dataset  # noqa: E402

# key, file, loader-type, expected count, how-to-obtain
DATASETS = [
    ("DRQA",        "data/DRQA/drqa_at10.jsonl",       "drqa",        568,   "bundled"),
    ("PopQA",       "data/PopQA/test.json",            "popqa",       4267,  "download"),
    ("RetrievalQA", "data/RetrievalQA/eval_all.json",  "retrievalqa", 2785,  "download"),
    ("NQ",          "data/NQ/test.json",               "nq",          3610,  "fid"),
    ("TriviaQA",    "data/TQA/test.json",              "tqa",         11313, "fid"),
]

FID_OBTAIN = ("run the FiD pipeline (github.com/facebookresearch/FiD, get-data.sh) "
              "then: python scripts/download_data.py {ds} --fid_file <FiD>/open_domain_data/{sub}/test.json")


def _download(dl_arg):
    subprocess.run([sys.executable, os.path.join(HERE, "download_data.py"), dl_arg],
                   check=True, cwd=ROOT)


def verify_one(name, rel, dtype, expected):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        return "SKIP", "not present"
    try:
        ds = load_dataset(path, dtype)
    except Exception as e:
        return "FAIL", f"load error: {type(e).__name__}: {str(e)[:60]}"
    n = len(ds)
    if n != expected:
        return "FAIL", f"count {n} != expected {expected}"
    ex = ds[0]
    if not ex.documents:
        return "FAIL", "first example has no documents/ctxs"
    if not getattr(ex, "answers", None):
        return "FAIL", "first example has no answers"
    return "PASS", f"{n} examples, {ex.doc_size} ctxs in ex0"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true",
                    help="fetch PopQA + RetrievalQA from Hugging Face before verifying")
    ap.add_argument("--fid_dir", default=None,
                    help="FiD open_domain_data dir; reformat NQ + TriviaQA from it")
    args = ap.parse_args()

    if args.download:
        print("== Downloading PopQA + RetrievalQA from their original HF releases ==")
        _download("popqa")
        _download("retrievalqa")
    if args.fid_dir:
        print("== Reformatting NQ + TriviaQA from FiD open_domain_data ==")
        for ds, sub in [("nq", "NQ"), ("tqa", "TQA")]:
            fid_file = os.path.join(args.fid_dir, sub, "test.json")
            if os.path.exists(fid_file):
                subprocess.run([sys.executable, os.path.join(HERE, "download_data.py"),
                                ds, "--fid_file", fid_file], check=True, cwd=ROOT)
            else:
                print(f"  {ds}: {fid_file} not found — skipping")

    # DRQA @5 is derived from @10 if missing.
    at5 = os.path.join(ROOT, "data/DRQA/drqa_at5.jsonl")
    if not os.path.exists(at5):
        subprocess.run([sys.executable, os.path.join(ROOT, "drqa", "derive_at5.py"),
                        "--in", os.path.join(ROOT, "data/DRQA/drqa_at10.jsonl"),
                        "--out", at5], check=True)

    print(f"\n{'dataset':<13} {'status':<6} detail")
    print("-" * 66)
    any_fail = False
    for name, rel, dtype, expected, obtain in DATASETS:
        status, detail = verify_one(name, rel, dtype, expected)
        any_fail |= (status == "FAIL")
        print(f"{name:<13} {status:<6} {detail}")
        if status == "SKIP" and obtain == "fid":
            sub = "NQ" if dtype == "nq" else "TQA"
            print(f"{'':<13}        -> {FID_OBTAIN.format(ds=dtype, sub=sub)}")
        elif status == "SKIP" and obtain == "download":
            print(f"{'':<13}        -> python scripts/download_data.py {dtype}  "
                  f"(or re-run with --download)")
    print("-" * 66)
    print("DRQA @5 derivation: ok" if os.path.exists(at5) else "DRQA @5: MISSING")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
