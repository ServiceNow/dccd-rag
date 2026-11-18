#!/usr/bin/env python3
"""Convert the original RetrievalQA release (HF ``aialt/RetrievalQA``) into the
DCCD QA format ``{question, answers, _id, data_source, ctxs:[{id,title,text}]}``.

Sources within RetrievalQA: freshqa(54) realtimeqa(188) toolqa(75) popqa(1570)
triviaqa(898) — 2,785 examples total (Zhang et al., 2024).

Two outputs match the bundled files:
  --all                 -> data/RetrievalQA/eval_all.json   (all 2,785, order preserved)
  (stratified, default) -> data/RetrievalQA/eval100_strat.json (100, seeded sample)
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

from huggingface_hub import hf_hub_download


def normalize_doc(doc, idx, qid):
    if isinstance(doc, str):
        return {"id": f"{qid}_d{idx}", "title": "", "text": doc}
    return {
        "id": doc.get("id") or f"{qid}_d{idx}",
        "title": doc.get("title") or "",
        "text": doc.get("text") or doc.get("content") or "",
    }


def convert_row(r, max_docs):
    qid = r["question_id"]
    raw = r.get("context") or []
    ctxs = [normalize_doc(d, i, qid) for i, d in enumerate(raw[:max_docs])]
    return {
        "question": r["question"],
        "answers": r["ground_truth"],
        "_id": qid,
        "data_source": r["data_source"],
        "param_knowledge_answerable": r.get("param_knowledge_answerable"),
        "ctxs": ctxs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/RetrievalQA/eval100_strat.json")
    p.add_argument("--max_docs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all", action="store_true",
                   help="convert ALL 2,785 examples (order preserved) -> eval_all.json")
    # Per-source sample sizes (stratified-100 plan; ignored with --all)
    p.add_argument("--n_freshqa", type=int, default=20)
    p.add_argument("--n_realtimeqa", type=int, default=30)
    p.add_argument("--n_toolqa", type=int, default=20)
    p.add_argument("--n_popqa", type=int, default=15)
    p.add_argument("--n_triviaqa", type=int, default=15)
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    path = hf_hub_download("aialt/RetrievalQA", "retrievalqa.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(path) if l.strip()]

    if args.all:
        sampled = rows
    else:
        by_src = defaultdict(list)
        for r in rows:
            by_src[r["data_source"]].append(r)
        rng = random.Random(args.seed)
        targets = {"freshqa": args.n_freshqa, "realtimeqa": args.n_realtimeqa,
                   "toolqa": args.n_toolqa, "popqa": args.n_popqa,
                   "triviaqa": args.n_triviaqa}
        sampled = []
        for src, n in targets.items():
            pool = by_src.get(src, [])
            if not pool:
                continue
            idx = sorted(rng.sample(range(len(pool)), min(n, len(pool))))
            sampled.extend(pool[i] for i in idx)
        rng.shuffle(sampled)

    converted = [convert_row(r, args.max_docs) for r in sampled]
    with open(args.out, "w") as f:
        json.dump(converted, f, ensure_ascii=False)

    nd = [len(c["ctxs"]) for c in converted]
    print(f"Wrote {len(converted)} examples to {args.out}")
    print(f"  source mix: {dict(Counter(c['data_source'] for c in converted))}")
    print(f"  docs/q: min={min(nd)} max={max(nd)} mean={sum(nd)/len(nd):.1f}")


if __name__ == "__main__":
    main()
