#!/usr/bin/env python3
"""Derive the DRQA @5 bundle from the DRQA @10 bundle.

Both DRQA configurations are derived from the *same* source examples (see the
paper, App. A "Final Dataset"). The @10 bundle contains, per example:

    1 correct, 2 misinfo, 2 temporal, 5 noise   (10 documents)

The @5 bundle keeps a strict per-example subset with the composition:

    1 correct, 1 misinfo, 1 temporal, 2 noise    (5 documents)

This script selects that subset deterministically (stable order within each
type, capped at the @5 counts) and re-shuffles the kept documents with a fixed
per-example seed so document position is not informative. A few examples have
fewer than the nominal number of a given type; we keep min(available, target),
mirroring the paper's note that a small number of examples fall below exact-k.

Usage:
    python drqa/derive_at5.py --in data/DRQA/drqa_at10.jsonl --out data/DRQA/drqa_at5.jsonl
"""
import argparse
import json
import random

# Target @5 composition (paper, Table: DRQA bundle configurations).
AT5_COUNTS = {"correct": 1, "misinfo": 1, "temporal": 1, "noise": 2}
SEED = 42


def derive_example(ex: dict, rng: random.Random) -> dict:
    by_type: dict[str, list] = {}
    for doc in ex["documents"]:
        by_type.setdefault(doc.get("type", "noise"), []).append(doc)

    kept = []
    for dtype, target in AT5_COUNTS.items():
        docs = by_type.get(dtype, [])
        kept.extend(docs[:target])

    # Any document whose type is not in the @5 schema is dropped. Re-shuffle so
    # the correct document is not at a fixed position.
    rng.shuffle(kept)
    out = dict(ex)
    out["documents"] = kept
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/DRQA/drqa_at10.jsonl")
    ap.add_argument("--out", dest="out", default="data/DRQA/drqa_at5.jsonl")
    args = ap.parse_args()

    n = 0
    with open(args.inp) as fin, open(args.out, "w") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            rng = random.Random(SEED + i)
            fout.write(json.dumps(derive_example(ex, rng), ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} examples to {args.out}")


if __name__ == "__main__":
    main()
