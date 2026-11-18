#!/usr/bin/env bash
# Prepare + verify datasets. DRQA ships with the repo; the standard-QA sets are
# obtained from their original sources. This is a thin wrapper around
# scripts/verify_data.py (which derives DRQA @5, fetches PopQA + RetrievalQA with
# --download, and checks every dataset's count + schema).
#
#   bash scripts/prepare_data.sh                       # verify what's present (+ derive DRQA @5)
#   bash scripts/prepare_data.sh --download            # also fetch PopQA + RetrievalQA
#   bash scripts/prepare_data.sh --download --fid_dir <FiD>/open_domain_data   # + NQ + TriviaQA
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/verify_data.py "$@"
