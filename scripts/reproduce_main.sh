#!/usr/bin/env bash
# Reproduce the main results (paper Table 1) for one model:
#   5 datasets x {@5, @10} x {full, cad, adacad, dvd, cocoa, dccd}  (+ zero-shot once per dataset)
#
# Usage:
#   bash scripts/reproduce_main.sh [MODEL_KEY] [OUT_ROOT]
#     MODEL_KEY  qwen3.5-2b | qwen3.5-9b | phi3-medium   (default: qwen3.5-2b)
#     OUT_ROOT   results directory                         (default: results)
#
#   MAX_EXAMPLES=20 bash scripts/reproduce_main.sh ...   quick smoke run
#   DRY_RUN=1       bash scripts/reproduce_main.sh ...   print the commands only
#
# Evaluate afterwards with:
#   python scripts/evaluate.py --metric strem <OUT_ROOT>/<model>/nq_at10/*.json     (NQ/TQA/PopQA/RetrievalQA)
#   python scripts/evaluate.py --metric judge <OUT_ROOT>/<model>/drqa_at5/*.json    (DRQA; needs OPENROUTER_API_KEY)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_KEY="${1:-qwen3.5-2b}"
OUT_ROOT="${2:-results}"

case "$MODEL_KEY" in
  qwen3.5-2b)  MODEL="Qwen/Qwen3.5-2B" ;;
  qwen3.5-9b)  MODEL="Qwen/Qwen3.5-9B" ;;
  phi3-medium) MODEL="microsoft/Phi-3-medium-128k-instruct" ;;
  *) echo "unknown MODEL_KEY '$MODEL_KEY' (expected qwen3.5-2b, qwen3.5-9b or phi3-medium)" >&2
     exit 1 ;;
esac

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
  # Zero-shot reference: documents are unused, so run it once per dataset.
  run python scripts/run_decoding.py --model "$MODEL" --dataset "$ds" \
    --data_file "$(data_file "$ds" 5)" --doc_num 5 --mode zeroshot \
    --out "$OUT_ROOT/$MODEL_KEY/$ds/zeroshot.json" ${EXTRA[@]+"${EXTRA[@]}"}
  for k in 5 10; do
    for m in full cad adacad dvd cocoa dccd; do
      run python scripts/run_decoding.py --model "$MODEL" --dataset "$ds" \
        --data_file "$(data_file "$ds" "$k")" --doc_num "$k" --mode "$m" \
        --out "$OUT_ROOT/$MODEL_KEY/${ds}_at${k}/$m.json" ${EXTRA[@]+"${EXTRA[@]}"}
    done
  done
done

echo "Done. Predictions are under $OUT_ROOT/$MODEL_KEY/. Evaluate them with scripts/evaluate.py." >&2
