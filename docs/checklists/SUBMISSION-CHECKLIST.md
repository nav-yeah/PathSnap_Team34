# Final submission checklist

The official submission has two deliverables: the public GitHub repository and one consolidated technical report PDF.

| Deliverable item | Repository evidence | Status / report action |
|---|---|---|
| Source, dependencies, README, model artifact | Root README, `requirements-dev.txt`, `artifacts/model.joblib` | Present on `main`; keep the repository public |
| Dataset exploration, preprocessing, and features | `docs/report-material/`; `results/final/evidence/` | Describe the actual training-only audit and proxy-label assumptions |
| Five-model results and Bayesian optimization | `results/review1/`; `results/final/evidence/outer.json` and trial records | Include the measured comparison and explain why the untuned HGB baseline was retained |
| HMM/Viterbi and classifier comparison | `pathsnap_ml/viterbi.py`; `results/final/evidence/final-routes.json`; `results/final/evaluator-results.json` | Include student and evaluator figures with the proxy-label caveat |
| Route reconstruction and continuity | `tests/test_viterbi.py`; route quality returned by API | Explain bridge insertion, explicit route breaks, and continuity limitations |
| Folium map and API/UI | `ui/`; `docs/DEPLOYMENT.md` | Insert a genuine map view and screenshots from the working two-laptop demo |
| Full test evidence | `results/final/evidence/stitching-tests.log` | 47 tests passed in the recorded run |
| One consolidated report PDF | Prepared by the team | Remaining submission task; check claims against `main` and disclose AI assistance as required |

Do not commit the raw university dataset, credentials, or data-bearing map exports. Do not use evaluator results for tuning or model selection.
