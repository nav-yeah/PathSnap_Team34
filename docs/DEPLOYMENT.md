# PathSnap — API, UI and two-laptop deployment

Owner: Aryan (Deployment Lead). Code: [`api/`](../api), [`ui/`](../ui), [`deploy/`](../deploy), [`tests/`](../tests).

```
 Laptop 2 (UI)                                 Laptop 1 (model + API)
 ┌──────────────────────┐   POST /predict     ┌────────────────────────────────────────────┐
 │ Streamlit  ui/app.py │ ──{"trip_id": …}──▶ │ FastAPI  api/main.py                        │
 │  • lists 10 trips    │                     │  trip_store  .track/.nodes/.arcs only       │
 │  • Folium map        │ ◀── route, metrics ─│  pipeline    candidates → model → decoder   │
 │  • no model, no data │     (aggregates)    │  evaluation  .route → aggregate scores only │
 └──────────────────────┘                     └────────────────────────────────────────────┘
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Model loaded?, model version, decoder in use, number of trips |
| GET | `/trips` | `trip_id`, observation count, duration for each available trip |
| POST | `/predict` | Body `{"trip_id": "00000052"}`. Returns the predicted arc per GPS point, a stitched route with explicit break indices, route continuity, server-side metrics and map geometry |
| GET | `/docs` | Interactive OpenAPI docs (useful for the "API response" screenshot) |

Error codes: `404` unknown trip, `422` malformed id / extra fields / corrupt trip files, `503` model artifact missing or incompatible, `500` pipeline or decoder failure (server stays up and logs the traceback).

## Ground-truth isolation (guideline: labels stay server-side)

* `api/trip_store.py` is the only loader used for inference and reads **only** `.track`, `.nodes`, `.arcs`.
* `api/evaluation.py` is the only code that opens `.route`. It runs **after** the prediction is final and returns aggregate numbers only (Top-1, candidate recall, label coverage, route-arc precision/recall). No reference arc ids, geometry or per-point correct/incorrect flags are returned.
* `PATHSNAP_REFERENCE_DIR` can point at a separate folder so `.route` files never sit next to inference inputs.
* Tests prove it: predictions are byte-identical with and without reference data; the inference path never calls `loadtxt` on a `.route`; the response schema is a fixed whitelist; the `predict` body rejects extra fields.

## Metrics shown in the UI

| Metric | Definition | Needs reference? |
|---|---|---|
| Top-1 accuracy | Decoded arc = reference-proxy arc, over confidently labelled points (team alignment rule: 35 m radius, 8 m margin). Candidate misses count as errors. | yes |
| Classifier-only Top-1 | Same, using the raw classifier argmax (for the HMM-vs-classifier comparison) | yes |
| Candidate recall | Reference arc is among the 8 candidates. **An upper bound, not model accuracy.** | yes |
| Label coverage | Share of GPS points with a confident label (rest excluded) | yes |
| Route arc precision / recall | Distinct predicted arcs vs distinct arcs in the `.route` file | yes |
| Route continuity | Share of arc changes where the next arc's `from_node` is the previous arc's `to_node` (drivable on the directed graph) | no |
| Disconnected jumps | Arc changes that are not connected | no |
| Inserted bridge arcs | Shortest directed road arcs inserted between decoded observations when reachable within 2 km; never inserted across a GPS gap or unmatched observation | no |
| Route breaks | Starts of a new route component after an unmatched observation, GPS gap or unreachable transition | no |

## Configuration (environment variables on Laptop 1)

| Variable | Default | Meaning |
|---|---|---|
| `PATHSNAP_MODEL` | `artifacts/model.joblib` | Team artifact following the `pathsnap-ml-v1` contract. Trusted path only; never load uploaded files |
| `PATHSNAP_TRIPS_DIR` | `data/eval_trips` | `<dir>/<trip_id>/<trip_id>.{track,nodes,arcs}` |
| `PATHSNAP_REFERENCE_DIR` | = trips dir | `<dir>/<trip_id>/<trip_id>.route` for scoring |
| `PATHSNAP_DECODER` | `classifier_argmax` | Or `package.module:function` (the team HMM) |
| `PATHSNAP_CORS_ORIGINS` | `*` | Comma-separated origins |

## Final model: Histogram Gradient Boosting

The team selected `HistGradientBoostingClassifier` (Review 1 comparison). The API is model-agnostic: anything following the `pathsnap-ml-v1` contract loads. Swapping models means changing the artifact file, not the code.

**Install the exact frozen artifact.** Nested tuning did not improve held-out outer-fold Top-1, so use the team-verified baseline refit, not a new model. The required SHA-256 is `9c2acba05bfa1922a412152525dfee624e67ca0845de62c262a424b56d9579a7`.

```powershell
python deploy/promote_model.py C:\trusted\model.joblib --dest artifacts\model.joblib
```

The script refuses the artifact if it isn't Histogram Gradient Boosting, if its schema or features differ, if scikit-learn reports a version mismatch with the training laptop, or if it can't score a candidate row through the same `predict()` the API uses. It prints the SHA-256 fingerprint.

**Proving which model is serving.** `/health`, the uvicorn startup log and the UI sidebar all show:

| Field | Example (untuned baseline) |
|---|---|
| family / estimator | Histogram Gradient Boosting / `HistGradientBoostingClassifier` |
| key_parameters | `learning_rate=0.1, max_iter=70, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0` |
| fitted_iterations | boosting rounds actually fitted (early stopping can stop before `max_iter`) |
| sha256 | fingerprint of the loaded `.joblib`. Compare with the verified digest above |
| version_warnings | scikit-learn version mismatches detected while unpickling (should be empty) |

Notes:
* `early_stopping="auto"` turns on for more than 10,000 rows, holding out 10% of the training rows internally. Report `fitted_iterations` alongside `max_iter`.
* HGB scores are independent per candidate (not normalised within a GPS point). The HMM should treat them as described in `pathsnap_ml/README.md` (clip, then log).
* The verified frozen model is committed at `artifacts/model.joblib`. Confirm its SHA-256 before the demonstration; use `deploy/promote_model.py` only when installing a separately supplied trusted copy.

## Decoder plug-in (for Navya's HMM/Viterbi)

```python
def decode(scores, candidate_arcs, context) -> np.ndarray:   # shape (N,), int arc ids, -1 = unmatched
    # scores          (N, 8) independent binary classifier scores in [0, 1]
    # candidate_arcs  (N, 8) arc ids aligned with scores
    # context.track (N,3) lon/lat/t · context.nodes · context.arcs (from,to)
    # context.points_m / starts_m / ends_m: trip-local metres (same frame as candidates)
```

Start the API with `PATHSNAP_DECODER=pathsnap_ml.viterbi:decode` (any importable path). The response then reports both decoded and classifier-only Top-1, which is the comparison the report asks for. A decoder that returns the wrong shape or out-of-range arc ids gives a 500 with a clear message; it does not crash the server.

## Demo-day runbook

**Both laptops, once** (Python 3.12 to match the team pins; I tested on 3.15.2 with the same package versions):

```powershell
git clone https://github.com/nav-yeah/PathSnap_Team34.git
cd PathSnap_Team34
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

**Laptop 1 (API)**

```powershell
python -m pip install -r api/requirements.txt
# put the final artifact at artifacts\model.joblib and the 10 evaluation trips in a folder
.\deploy\open_firewall.ps1               # once, in an Administrator PowerShell
.\deploy\run_api.ps1 -Model artifacts\model.joblib -Trips "C:\PathSnap\eval_trips" -Decoder pathsnap_ml.viterbi:decode
```

The script prints Laptop 1's IPv4 addresses. Keep this window visible: every request is logged with client IP, trip, Top-1, continuity and latency. Evaluators want to see these logs.

**Laptop 2 (UI)**

```powershell
python -m pip install -r ui/requirements.txt
.\deploy\run_ui.ps1 -Api http://<laptop1-ip>:8000
```

**Pre-flight check (from Laptop 2)**: `python deploy/smoke_test.py --api http://<laptop1-ip>:8000`. It calls every trip and exits non-zero if any fails.

macOS/Linux equivalents: `deploy/run_api.sh MODEL TRIPS_DIR [DECODER]`, `deploy/run_ui.sh http://<ip>:8000`.

### Troubleshooting

| Symptom | Fix |
|---|---|
| UI: "Cannot reach the API" | uvicorn must use `--host 0.0.0.0` (the scripts do). Both laptops on the same Wi-Fi / hotspot. Test `http://<ip>:8000/health` in Laptop 2's browser |
| UI: "Timed out connecting" | Windows firewall: run `open_firewall.ps1`; network profile must be *Private*. Campus Wi-Fi often blocks client-to-client traffic: use a phone hotspot |
| `/health` degraded, "Incompatible artifact schema" | Artifact built with a different feature schema; restore the approved frozen artifact and use the pinned dependencies |
| `/health` degraded, unpickling error | scikit-learn version differs from the training laptop; install `pathsnap_ml/requirements.txt` exactly |
| `trips_available: 0` | `PATHSNAP_TRIPS_DIR` must contain `<trip_id>/<trip_id>.track/.nodes/.arcs` folders |

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests pathsnap_ml/test_pipeline.py -q
```

The `tests/` suite uses a synthetic two-road network, so no university data is needed. It covers all endpoints, error codes, path traversal, missing or incompatible artifacts, corrupt files, decoder plug-ins, ground-truth isolation, client error messages, the Folium layers, and a headless Streamlit run (`streamlit.testing`).

Map export for the report: `python -m ui.map_view --api http://<ip>:8000 --trip 00000052 --out map.html`.
