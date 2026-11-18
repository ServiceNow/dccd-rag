#!/usr/bin/env python3
"""Obtain the exact evaluation datasets from their original sources.

Only DRQA ships with the repository; this script fetches the standard QA
datasets from their upstream releases into ``data/``. See data/README.md for
sources and citations.

Subcommands:
  popqa        Download HF ``Atipico1/popQA_preprocessed`` (test) -> data/PopQA/test.json
  retrievalqa  Convert HF ``aialt/RetrievalQA`` -> data/RetrievalQA/{eval_all,eval100_strat}.json
  nq | tqa     Reformat an FiD ``open_domain_data`` reader file (ctxs -> {id,title,text})
               -> data/{NQ,TQA}/test.json.  Requires --fid_file (see below).

NQ / TriviaQA come from the DPR/Fusion-in-Decoder pipeline (Izacard & Grave 2021).
The {question, answers, ctxs[100]} reader files are produced by the FiD repo's
``get-data.sh`` (downloads NQ-open / TriviaQA + DPR ``psgs_w100`` + DPR retrieval,
then ``src/preprocess.py``). Run that pipeline, then point ``--fid_file`` at the
resulting ``open_domain_data/{NQ,TQA}/test.json`` to strip each ctx to the
bundled ``{id,title,text}`` schema.
  FiD: https://github.com/facebookresearch/FiD  (get-data.sh)
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _out(*parts):
    p = os.path.join(ROOT, "data", *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def download_popqa():
    from datasets import load_dataset
    ds = load_dataset("Atipico1/popQA_preprocessed", split="test")
    rows = [dict(r) for r in ds]
    out = _out("PopQA", "test.json")
    with open(out, "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"PopQA: wrote {len(rows)} examples -> {out}")
    print("  (original release: Mallen et al. 2023; retrieval-augmented release "
          "Atipico1/popQA_preprocessed test split)")


def download_retrievalqa():
    conv = os.path.join(HERE, "convert_retrievalqa.py")
    subprocess.run([sys.executable, conv, "--all", "--out", _out("RetrievalQA", "eval_all.json")],
                   check=True)
    subprocess.run([sys.executable, conv, "--out", _out("RetrievalQA", "eval100_strat.json")],
                   check=True)


def reformat_fid(fid_file, dataset):
    """Strip an FiD reader file's ctxs to {id,title,text} (bundled schema)."""
    data = json.load(open(fid_file))
    out_rows = []
    for ex in data:
        row = {"question": ex["question"], "answers": ex["answers"]}
        if "target" in ex:
            row["target"] = ex["target"]
        row["ctxs"] = [{"id": c.get("id"), "title": c.get("title"), "text": c.get("text")}
                       for c in ex.get("ctxs", [])]
        out_rows.append(row)
    out = _out("NQ" if dataset == "nq" else "TQA", "test.json")
    with open(out, "w") as f:
        json.dump(out_rows, f, ensure_ascii=False)
    print(f"{dataset.upper()}: reformatted {len(out_rows)} examples -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["popqa", "retrievalqa", "nq", "tqa"])
    ap.add_argument("--fid_file", help="(nq/tqa) path to the FiD open_domain_data test.json")
    args = ap.parse_args()

    if args.dataset == "popqa":
        download_popqa()
    elif args.dataset == "retrievalqa":
        download_retrievalqa()
    else:
        if not args.fid_file:
            sys.exit("nq/tqa require --fid_file (the FiD open_domain_data test.json; "
                     "produce it with the FiD repo get-data.sh — see this file's header).")
        reformat_fid(args.fid_file, args.dataset)


if __name__ == "__main__":
    main()
