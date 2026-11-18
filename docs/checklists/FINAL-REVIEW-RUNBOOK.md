# Final demo runbook

Use the exact model committed at `artifacts/model.joblib`; do not retrain for the demonstration. The expected SHA-256 is `9c2acba05bfa1922a412152525dfee624e67ca0845de62c262a424b56d9579a7`.

## Laptop 1 — API

Place each of the ten trips in its own folder with `.track`, `.nodes`, `.arcs`, and `.route` files. From the repository root, open PowerShell and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\deploy\run_api.ps1 `
  -Model artifacts\model.joblib `
  -Trips C:\trips\evaluator_test `
  -References C:\trips\evaluator_test `
  -Decoder pathsnap_ml.viterbi:decode `
  -Port 8000
```

If needed, allow port 8000 through Windows Firewall on a private network from an Administrator PowerShell:

```powershell
.\deploy\open_firewall.ps1 -Port 8000
```

Give Laptop 2 an active IPv4 address printed by the server launcher. Keep the server window open so the request logs are visible.

## Laptop 2 — interface

```powershell
.\deploy\run_ui.ps1 -Api http://<LAPTOP1-IP>:8000
```

Open <http://localhost:8501>, select a trip, and reconstruct its route. Verify health and run the ten-trip smoke test from Laptop 2:

```powershell
Invoke-RestMethod http://<LAPTOP1-IP>:8000/health
py -3.12 deploy\smoke_test.py --api http://<LAPTOP1-IP>:8000 --out smoke-results.json
```

## Evidence for the report

Capture a screenshot of the interface with the map and results, and a screenshot of Laptop 1 showing requests from Laptop 2. Include the actual smoke-test output and report the model fingerprint. The API evaluates reference routes server-side after prediction; do not expose per-observation labels in the UI.
