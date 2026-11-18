#!/usr/bin/env bash
# Reproduce the ablation study (paper Table 2):
#   Qwen3.5-2B x 5 datasets x {@5, @10} x {dccd, fixed_gate, token_only, doc_only, random}
#
# Usage:
#   bash scripts/reproduce_ablation.sh [OUT_ROOT]          (default OUT_ROOT: results/ablation)
#   MAX_EXAMPLES=20 bash scripts/reproduce_ablation.sh   quick smoke run
#   DRY_RUN=1       bash scripts/reproduce_ablation.sh   print the commands only
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_ROOT="${1:-results/ablation}"
MODEL="Qwen/Qwen3.5-2B"

run() { if [[ -n "${DRY_RUN:-}" ]]; then echo "$*"; else "$@"; fi; }

data_file() {  # data_file DATASET DEPTH
  case "$1" in
    drqa)        echo "data/DRQA/drqa_at$2.jsonl" ;;
    nq)          echo "data/NQ/test.json" ;;
    tqa)         echo "data/TQA/test.json" ;;
    retrievalqa) echo "data/RetrievalQA/eval_all.json" ;;
    popqa)       echo "data/PopQA/test.json" ;;
  esac
}

EXTRA=()
if [[ -n "${MAX_EXAMPLES:-}" ]]; then EXTRA=(--max_examples "$MAX_EXAMPLES"); fi

# Only DRQA @10 ships; derive the @5 bundle from it if absent.
if [[ ! -f data/DRQA/drqa_at5.jsonl ]]; then
  run python drqa/derive_at5.py --in data/DRQA/drqa_at10.jsonl --out data/DRQA/drqa_at5.jsonl
fi

for ds in drqa nq tqa retrievalqa popqa; do
  for k in 5 10; do
    for m in dccd fixed_gate token_only doc_only random; do
      run python scripts/run_decoding.py --model "$MODEL" --dataset "$ds" \
        --data_file "$(data_file "$ds" "$k")" --doc_num "$k" --mode "$m" \
        --out "$OUT_ROOT/${ds}_at${k}/$m.json" ${EXTRA[@]+"${EXTRA[@]}"}
    done
  done
done

echo "Done. Ablation predictions are under $OUT_ROOT/." >&2
