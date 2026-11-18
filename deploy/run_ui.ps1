# Laptop 2 — start the evaluator UI (run from the repo root).
#   .\deploy\run_ui.ps1 -Api http://192.168.1.20:8000
param([Parameter(Mandatory = $true)][string]$Api, [int]$Port = 8501)
$env:PATHSNAP_API_URL = $Api
python -m streamlit run ui\app.py --server.port $Port --server.headless false
