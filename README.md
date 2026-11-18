# PathSnap — Team 34

PathSnap matches noisy GPS observations to candidate road segments, scores the candidates with a frozen Histogram Gradient Boosting classifier, and reconstructs a connected route with Viterbi decoding and directed route stitching.

**Repository:** <https://github.com/nav-yeah/PathSnap_Team34>

## Project status

The five-model comparison, Bayesian tuning, saved model, API, UI, Viterbi decoder, route stitching, evaluator run, and two-laptop setup are complete. The remaining submission deliverable is the consolidated technical report PDF.

The model was selected from grouped cross-validation. Fifteen Optuna TPE trials were run; tuning did not improve the baseline, so the baseline HGB model was retained.

## Results

Accuracy values are agreement with derived reference-route proxy labels, not independently verified observation-level ground truth.

| Partition | Classifier-only Top-1 | HGB + Viterbi Top-1 | Accepted observations |
|---|---:|---:|---:|
| Ten student-test trips | 86.32% | 91.92% | 13,006 |
| Ten evaluator trips | 75.19% | 83.05% | 14,045 |

The evaluator run covered 19,674 observations and had approximately 99.7% candidate recall. Route stitching inserted 906 bridge arcs and recorded 5 route breaks at GPS gaps; there were no disconnected jumps within stitched components. Continuity is measured within components and is partly guaranteed by stitching, so it is not an independent route-accuracy measure.

Evaluator results are final evaluation evidence only. They were not used for model selection or hyperparameter tuning. The aggregated per-trip output is in [`results/final/evaluator-results.json`](results/final/evaluator-results.json).

## Repository layout

```text
api/                         FastAPI service and inference pipeline
artifacts/model.joblib       Frozen HGB model
deploy/                       API/UI launchers, firewall helper, smoke test
docs/                         Deployment, handoff, report material, checklists
  archive/                    Historical Review 1 snapshots and superseded handoffs
  checklists/                 Current submission and demonstration checklists
  report-material/            Dataset audit and report-ready sections
pathsnap_ml/                  Candidate/model interfaces, training, Viterbi decoder
results/                      Experiment metrics, validation records, final evidence
  review1/                    Model comparison and Review 1 outputs
  final/                      Final evaluation and verification evidence
scripts/                      Standalone audit, evaluation, and plotting scripts
tests/                        API, UI, decoder, and route-stitching tests
ui/                           Streamlit evaluator UI and Folium map
```

The university dataset is not included. Keep the `.track`, `.nodes`, `.arcs`, and `.route` files on the evaluation server and do not commit them.

## Setup

Use Python 3.12. From the repository root, create an environment and install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.12 -m pip install -r requirements-dev.txt
```

Verify the committed model fingerprint:

```powershell
Get-FileHash artifacts\model.joblib -Algorithm SHA256
```

Expected SHA-256:

```text
9c2acba05bfa1922a412152525dfee624e67ca0845de62c262a424b56d9579a7
```

## Trip data layout

Extract the ten trips on Laptop 1 so each trip has its own folder:

```text
C:\trips\evaluator_test\
├── 00000015\
│   ├── 00000015.track
│   ├── 00000015.nodes
│   ├── 00000015.arcs
│   └── 00000015.route
├── 00000017\
│   └── ...
└── ...
```

Prediction uses `.track`, `.nodes`, and `.arcs`. The reference `.route` is read for aggregate scoring after prediction; reference arc IDs and per-observation correctness are not sent to the UI.

## Two-laptop demo

Both laptops must be on the same Wi-Fi network or hotspot. Run commands from the repository root.

### Laptop 1 — API and model

If needed, allow inbound API traffic on the private network from an Administrator PowerShell:

```powershell
.\deploy\open_firewall.ps1 -Port 8000
```

In a normal PowerShell window, start the API. Replace the trip path if your extracted folder differs:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\deploy\run_api.ps1 `
  -Model artifacts\model.joblib `
  -Trips C:\trips\evaluator_test `
  -References C:\trips\evaluator_test `
  -Decoder pathsnap_ml.viterbi:decode `
  -Port 8000
```

Give Laptop 2 one of the IPv4 addresses printed by the script, for example `http://10.30.201.254:8000`. Leave the API window open to show incoming request logs.

### Laptop 2 — UI

Start the UI using Laptop 1's address:

```powershell
.\deploy\run_ui.ps1 -Api http://<LAPTOP1-IP>:8000
```

Open <http://localhost:8501>, select a trip, and click **Reconstruct route**.

### Verify the connection and all trips

From Laptop 2, check the API health and run all trips through the smoke test:

```powershell
Invoke-RestMethod http://<LAPTOP1-IP>:8000/health
py -3.12 deploy\smoke_test.py `
  --api http://<LAPTOP1-IP>:8000 `
  --out smoke-results.json
```

The smoke test exits successfully when every listed trip returns a successful response. To verify unknown-trip handling, submit an unknown ID and confirm the API returns HTTP 404:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://<LAPTOP1-IP>:8000/predict `
  -ContentType 'application/json' `
  -Body '{"trip_id":"99999999"}'
```

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Model, decoder, and service status |
| `GET` | `/trips` | Available trip IDs and observation counts |
| `POST` | `/predict` | Predict and reconstruct a selected trip |
| `GET` | `/docs` | Interactive API documentation |

Example request:

```json
{"trip_id": "00000052"}
```

## Tests and map export

Run the automated suite from the repository root:

```powershell
py -3.12 -m pytest -q
```

The committed full-suite log records 47 passing tests at the time of that run: [`results/final/evidence/stitching-tests.log`](results/final/evidence/stitching-tests.log).

Export a Folium map while the API is running:

```powershell
py -3.12 -m ui.map_view `
  --api http://localhost:8000 `
  --trip 00000052 `
  --out map.html
```

## Data and metric notes

- The `.route` files contain an ordered sequence of reference arc IDs; they do not directly map each GPS observation to an arc. The project aligns observations to the route and treats accepted labels as proxy labels.
- Top-1 is computed over accepted proxy labels. Candidate misses count as errors; uncertain alignments are excluded. Report label coverage and candidate recall alongside Top-1.
- Candidate recall means the reference proxy arc was present among generated candidates. It is not classifier accuracy.
- The classifier's candidate scores are compatibility estimates, not calibrated HMM likelihoods.
- Route stitching inserts a directed connector path only when one is reachable within the configured 2 km bound. It starts a new component across GPS gaps, unmatched observations, or unreachable transitions.
- The evaluator partition must not be used to tune models, thresholds, or decoder settings. Keep reference data server-side and do not expose labels through the UI.

## Submission

The submission consists of this public source repository and one consolidated technical report PDF. The report should use the committed experiment evidence, include the required maps and two-laptop screenshots, and describe proxy-label and route-continuity limitations.

For detailed deployment and troubleshooting guidance, see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). For an index of result files, see [`results/README.md`](results/README.md).
