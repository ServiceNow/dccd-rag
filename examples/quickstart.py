#!/usr/bin/env python3
"""Quickstart: full-context decoding vs. DCCD on one DRQA question.

By default this runs the example from the paper's appendix (DRQA qid 194, ten
retrieved documents: one correct, two misinformation, two outdated, five noise)
and prints both answers, DCCD's document-level confidence for every document,
and which document DCCD trusted (positive) and contrasted against (negative)
at every decoding step.

    python examples/quickstart.py                                   # Qwen3.5-2B, GPU recommended
    python examples/quickstart.py --model Qwen/Qwen2.5-0.5B-Instruct --device_map cpu --dtype float32
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dccd import DCCD  # noqa: E402
from dccd.data import load_dataset  # noqa: E402


def _label(docs, i):
    return f"#{i + 1} {docs[i].doc_type or ''}".strip() if i is not None else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B", help="HF model id or local path")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "DRQA", "drqa_at10.jsonl"))
    ap.add_argument("--qid", type=int, default=194, help="DRQA example index")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=60)
    args = ap.parse_args()

    example = load_dataset(args.data, "drqa")[args.qid]
    docs = example.documents
    rag = DCCD(args.model, dtype=args.dtype, device_map=args.device_map,
               max_new_tokens=args.max_new_tokens)

    full = rag.generate(example.question, docs, mode="full")
    out = rag.generate(example.question, docs, mode="dccd", return_trace=True)

    print(f"\nQuestion: {example.question}")
    print(f"Gold:     {example.answers[0]}\n")
    print("Document-level confidence q (support probe):")
    best = max(range(len(docs)), key=lambda i: out["q_scores"][i])
    for i, q in enumerate(out["q_scores"]):
        mark = "  <- highest" if i == best else ""
        print(f"  {_label(docs, i):<14} q = {q:.3f}{mark}")
    print(f"\nfull-context : {full}")
    print(f"DCCD         : {out['prediction']}\n")
    print("DCCD per-step trace (positive = trusted stream, negative = contrasted stream):")
    print(f"  {'step':>4}  {'token':<16} {'positive':<14} {'negative':<14} {'eps':>5}")
    for t in out["trace"]:
        eps = f"{t['eps']:.3f}" if t["eps"] is not None else "-"
        print(f"  {t['step']:>4}  {t['token']!r:<16} {_label(docs, t['pos_doc']):<14} "
              f"{_label(docs, t['neg_doc']):<14} {eps:>5}")


if __name__ == "__main__":
    main()
