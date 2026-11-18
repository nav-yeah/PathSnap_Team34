# Final project readiness

The engineering implementation is merged to `main`. The ten-trip evaluator run and the two-laptop API/UI setup are complete. The remaining submission task is the consolidated technical report PDF, including a representative Folium map and genuine demonstration screenshots.

## Verified repository evidence

- Frozen Histogram Gradient Boosting model: `artifacts/model.joblib`; SHA-256 `9c2acba05bfa1922a412152525dfee624e67ca0845de62c262a424b56d9579a7`.
- Five-model comparison: `results/review1/integrated/`.
- Bayesian search and final model metadata: `results/final/evidence/outer.json`, trial records, and `model.json`.
- Student route comparison: `results/final/evidence/final-routes.json`.
- Evaluator aggregate: `results/final/evaluator-results.json`.
- Route stitching verification: `tests/test_viterbi.py` and `results/final/evidence/stitching-tests.log` (47 passed in the recorded full-suite run).
- Deployment instructions: `docs/DEPLOYMENT.md`.

## Report checks

1. Use the official report sections and consolidate the team contributions into one PDF.
2. State that observation labels are derived reference-route proxies, and report accepted-label coverage.
3. Keep student-test results separate from evaluator results. The evaluator partition was not used for tuning or selection.
4. Describe inserted bridge arcs and route breaks. Do not present within-component continuity as independent route accuracy.
5. Add the actual Folium map and screenshots from the working two-laptop demonstration.
6. Verify the report’s numbers and file references against the public `main` branch.
