#!/usr/bin/env python3
"""Evaluate prediction files written by ``run_decoding.py``.

* NQ / TriviaQA / PopQA / RetrievalQA  -> normalized string exact-match (str-em).
* DRQA                                 -> LLM-as-judge accuracy (OpenRouter).

Examples
--------
    # str-em over one or more QA prediction files
    python scripts/evaluate.py --metric strem results/q35_2b/nq_at10/*.json

    # LLM-judge a DRQA prediction file (needs OPENROUTER_API_KEY)
    python scripts/evaluate.py --metric judge \
        --judge_model anthropic/claude-opus-4-5 \
        results/q35_2b/drqa_at5/dccd.json

    # Inspect judge progress without calling the API
    python scripts/evaluate.py --metric judge --dry_run results/.../dccd.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dccd.eval import judge_drqa, score_str_em


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("preds", nargs="+", help="prediction JSON file(s)")
    p.add_argument("--metric", choices=["strem", "judge"], default="strem")
    p.add_argument("--full", action="store_true",
                   help="score the full prediction instead of its first sentence "
                        "(default: first sentence, the paper's protocol; both metrics)")
    p.add_argument("--judge_model", default="anthropic/claude-opus-4-5")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    try:  # read OPENROUTER_API_KEY etc. from a .env in the working directory (or above)
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass
    print(f"{'file':<70} {'n':>6} {'accuracy':>10}")
    for path in args.preds:
        with open(path) as f:
            records = json.load(f)
        name = os.path.basename(path)[:68]
        if args.metric == "strem":
            r = score_str_em(records, use_first_sentence=not args.full)
            print(f"{name:<70} {r['n']:>6} {r['accuracy']*100:>9.2f}%")
            continue
        cache = path + (".judge_full.jsonl" if args.full else ".judge.jsonl")
        r = judge_drqa(records, model=args.judge_model, cache_path=cache,
                       concurrency=args.concurrency, dry_run=args.dry_run,
                       use_first_sentence=not args.full)
        if r.get("dry_run"):
            print(f"{name:<70} {r['n']:>6}   dry-run "
                  f"(cached={r['cached']}, pending={r['pending']})")
            continue
        print(f"{name:<70} {r['n']:>6} {r['accuracy']*100:>9.2f}%")
        if r["errors"]:
            print(f"  WARNING: {r['errors']} prediction(s) could not be graded (API errors); "
                  f"they count as incorrect above and will be retried on the next run.")


if __name__ == "__main__":
    main()
