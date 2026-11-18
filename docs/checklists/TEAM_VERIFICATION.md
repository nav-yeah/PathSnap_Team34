# Team verification status

Engineering status after PR #16 was merged to `main`. This checklist records project implementation; the consolidated report PDF is a separate submission deliverable.

| Workstream | Status | Evidence |
|---|---|---|
| Dataset inspection, preprocessing, candidates, and proxy-label alignment | Complete | `docs/report-material/CHETHA_PREPROCESSING_VERIFICATION.md`; `results/final/evidence/` |
| Five-model comparison and grouped validation | Complete | `results/review1/integrated/`; `results/review1/` |
| Bayesian tuning and frozen HGB model | Complete | `results/final/evidence/outer.json`, trial records, `model.json`; `artifacts/model.joblib` |
| HMM/Viterbi decoding and directed route stitching | Complete | `pathsnap_ml/viterbi.py`; `tests/test_viterbi.py`; `results/final/evidence/stitching-tests.log` |
| Evaluator run on all ten trips | Complete | `results/final/evaluator-results.json` |
| FastAPI, Streamlit/Folium, and two-laptop integration | Working | `docs/DEPLOYMENT.md`; API smoke evidence under `results/final/evidence/` |
| Consolidated technical report PDF | Remaining submission task | Use the official report structure; include the actual results, map, limitations, and demonstration screenshots |

## Results to report carefully

All Top‑1 figures that use aligned route labels are proxy-label agreement. On the student-test set, classifier-only Top‑1 was 86.32% and HGB plus Viterbi was 91.92% (13,006 accepted observations). On the evaluator set, classifier-only Top‑1 was 75.19% and HGB plus Viterbi was 83.05% (14,045 accepted observations). Evaluator results are final evaluation only and were not used to tune or select the system.

The evaluator route post-processor inserted 906 directed bridge arcs and preserved five breaks at GPS gaps. Continuity within stitched components is partly guaranteed by construction; report breaks and proxy-label scores alongside it.
