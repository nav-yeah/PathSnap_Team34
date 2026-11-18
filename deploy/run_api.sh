#!/usr/bin/env bash
# Laptop 1 (macOS/Linux) — run from the repo root:
#   deploy/run_api.sh artifacts/model.joblib /path/to/eval_trips [decoder]
set -euo pipefail
export PATHSNAP_MODEL="${1:-artifacts/model.joblib}"
export PATHSNAP_TRIPS_DIR="${2:-data/eval_trips}"
export PATHSNAP_REFERENCE_DIR="${PATHSNAP_REFERENCE_DIR:-$PATHSNAP_TRIPS_DIR}"
export PATHSNAP_DECODER="${3:-classifier_argmax}"
PORT="${PORT:-8000}"
echo "Laptop 1 addresses:"
(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}') | sed "s|^|  http://|; s|$|:$PORT|"
exec python -m uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
