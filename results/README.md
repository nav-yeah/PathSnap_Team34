# Experiment and verification results

This folder contains recorded results and evidence. Run logs and tables are retained as generated; do not edit reported metric values by hand.

## Layout

- `review1/` — five-model comparison, fold manifests, teammate result bundles, and validation summaries.
- `final/evidence/` — final model metadata, tuning trials, student-test results, preprocessing evidence, API smoke outputs, and test logs.
- `final/evaluator-results.json` — aggregate per-trip evaluator output from the final frozen run. Evaluator data was not used to select or tune the system.

All accuracy figures that use alignment-derived labels are proxy-label agreement. Candidate recall is a candidate-generation metric, and stitched continuity is computed within route components. Neither should be described as observation-level ground-truth accuracy.
