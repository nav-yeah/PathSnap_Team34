#!/usr/bin/env bash
# Laptop 2 (macOS/Linux) — run from the repo root:  deploy/run_ui.sh http://<laptop1-ip>:8000
set -euo pipefail
export PATHSNAP_API_URL="${1:?usage: deploy/run_ui.sh http://<laptop1-ip>:8000}"
exec python -m streamlit run ui/app.py --server.port "${PORT:-8501}"
