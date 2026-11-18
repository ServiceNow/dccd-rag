#!/usr/bin/env python3
"""Run a decoding method over a dataset and write predictions.

Output is a JSON object ``{qid: {question, answers, prediction, ...}}`` — the
format consumed by ``scripts/evaluate.py``.

Examples
--------
    # DCCD on DRQA @5 with Qwen3.5-2B
    python scripts/run_decoding.py \
        --model Qwen/Qwen3.5-2B \
        --dataset drqa --data_file data/DRQA/drqa_at5.jsonl \
        --doc_num 5 --mode dccd \
        --out results/q35_2b/drqa_at5/dccd.json

    # Full-context baseline on NQ @10
    python scripts/run_decoding.py \
        --model Qwen/Qwen3.5-2B \
        --dataset nq --data_file data/NQ/test.json \
        --doc_num 10 --mode full --out results/q35_2b/nq_at10/full.json
"""
import argparse
import json
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dccd import ALL_MODES
from dccd.data import load_dataset
from dccd.decoding import GenConfig, generate
from dccd.models import load_model_and_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--dataset", required=True,
                   choices=["nq", "tqa", "popqa", "retrievalqa", "drqa"])
    p.add_argument("--data_file", required=True, help="path to the dataset file")
    p.add_argument("--mode", required=True, choices=list(ALL_MODES))
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--doc_num", type=int, default=5, help="retrieval depth (5 or 10)")
    p.add_argument("--max_new_tokens", type=int, default=60)
    p.add_argument("--max_length", type=int, default=16384,
                   help="max prompt+gen length; longer prompts are truncated to fit. Default 16384 "
                        "never truncates any bundled dataset (paper used 8000 for NQ/TQA/PopQA/"
                        "RetrievalQA and 16384 for DRQA; neither truncates, so 16384 reproduces all).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=10,
                   help="top-k for DCCD's token-level confidence (DVD always uses k=10)")
    p.add_argument("--no_meta", action="store_true",
                   help="format documents as bare text (drop 'Document [i]' prefix)")
    p.add_argument("--enable_thinking", action="store_true")
    p.add_argument("--no_chat_template", action="store_true")
    p.add_argument("--no_stop_first_sentence", action="store_true",
                   help="generate the full --max_new_tokens instead of stopping at the "
                        "first sentence. Default stops at the first sentence, which gives "
                        "the identical first_sentence(prediction) used by eval but faster; "
                        "use this flag to reproduce the raw full-length generations.")
    p.add_argument("--return_trace", action="store_true",
                   help="DCCD-family modes: store the per-step positive/negative document "
                        "and gate eps in each output record")
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--max_examples", type=int, default=None,
                   help="limit to the first N examples (smoke tests)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--trust_remote_code", action="store_true",
                   help="allow the checkpoint's own modelling code from the Hub "
                        "(not needed for the paper's models)")
    return p.parse_args()


def main():
    args = parse_args()
    examples = load_dataset(args.data_file, args.dataset)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    model, tokenizer = load_model_and_tokenizer(
        args.model, dtype=args.dtype, device_map=args.device_map,
        trust_remote_code=args.trust_remote_code)

    cfg = GenConfig(
        mode=args.mode,
        doc_num=args.doc_num,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        no_meta=args.no_meta,
        enable_thinking=args.enable_thinking,
        no_chat_template=args.no_chat_template,
        stop_first_sentence=not args.no_stop_first_sentence,
        return_trace=args.return_trace,
        random_seed=args.random_seed,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    results = {}
    for ex in tqdm(examples, desc=f"{args.mode}@{args.doc_num} {args.dataset}"):
        rec = generate(model, tokenizer, ex, cfg)
        results[rec["qid"]] = rec

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(results)} predictions to {args.out}")


if __name__ == "__main__":
    main()
