# Review 1 engineering evidence

> Archived Review 1 snapshot. Some status statements and paths describe an earlier repository state; use the root README and current checklists for the final project state.


`review1-comparison.json` is the output of the shared comparator for Chhavi's two untuned models. Results for Decision Tree, Random Forest and Histogram Gradient Boosting are pending teammate runs. No final model has been selected.

The run used Python 3.12.14, NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.9.1 and Optuna 5.1.0. Four tests passed under pytest 9.2.0. Both fitted artifacts were reloaded and checked against the original pipeline on reference-free candidate rows. Model binaries, per-trip evidence and the university archive are kept out of this publication.

See [the shared protocol](../../../pathsnap_ml/README.md#review-1-shared-untuned-protocol) to reproduce. The comparator reports equal-weight means of three trip-grouped folds on the 80 training trips, using reference-proxy labels rather than verified per-point truth. Top-1 includes candidate misses among accepted observations. Neither student-testing nor evaluator trips were used. The Optuna stage is separate; it was not run for these untuned baselines.
