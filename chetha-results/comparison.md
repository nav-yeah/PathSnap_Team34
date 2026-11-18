# Chetha Review 1 — Decision Tree vs Random Forest

> Results use the shared **reference-proxy** labels. The supplied route is documented as the chronological reference arc sequence, but it is not independently verified physical ground truth. These metrics are proxy-label evaluation, not verified ground-truth accuracy.

## Protocol used

- Ran the existing shared `pathsnap_ml.review1` baseline command on the 80 `train` trips only; `student_test` was not read.
- Preserved the shared adapter, ordered route-occurrence alignment, accepted-label rule (within 35 m and at least 8 m path-cost margin), eight candidates, binary candidate target, ranking rule, three GroupKFold folds, seed, and metric definitions.
- Features: `distance_m`, `heading_cos`, `fraction`, `length_m`, `speed_mps`. Identifiers and route labels are not predictors. Median imputation with missingness indicators is fit per training fold; tree models do not use scaling.
- These are untuned shared baselines. The configured Decision Tree uses `max_depth=8`; Random Forest uses 40 trees and `max_depth=10` (seed 34).
- All model runs used the same data SHA-256 and exact validation trip IDs as the saved team manifest. Fold parity with the legacy manifest: **yes**.

## Equal-weight mean across three validation folds

| Metric | Decision Tree | Random Forest |
|---|---:|---:|
| Candidate-row accuracy | 94.74% | 94.88% |
| Precision | 80.32% | 83.53% |
| Recall | 76.68% | 73.33% |
| Macro F1 | 87.71% | 87.60% |
| Weighted F1 | 94.69% | 94.74% |
| Observation-level Top-1 | 80.67% | 81.34% |
| Candidate recall | 99.56% | 99.56% |
| Candidate miss rate | 0.44% | 0.44% |
| Accepted-label coverage | 63.63% | 63.63% |

| Runtime | Decision Tree | Random Forest |
|---|---:|---:|
| Mean fit time per fold | 3.847 s | 23.192 s |
| Full-train artifact fit | 6.058 s | 36.642 s |
| Mean inference time per validation fold | 0.099 s | 0.921 s |
| Top-1 fold standard deviation | 1.76% | 1.86% |

## Interpretation

Under the shared selection metric (mean fold Top-1), Random Forest is higher by 0.67 percentage points. It also has slightly higher candidate-row accuracy. Decision Tree has slightly higher macro F1 and recall. These are close, untuned proxy-label baselines; no verified-ground-truth winner is established.

Candidate recall is candidate-set coverage of accepted proxy labels, not classifier accuracy. Top-1 counts a candidate miss as a failure; its denominator is accepted observations. The 63.63% coverage is before candidate ranking and excludes uncertain proxy labels.

## Comparison with Chhavi's saved results

The root `review1-comparison.json` lists Logistic Regression Top-1 80.34% and Gaussian Naive Bayes 79.03%. They are **not directly compared or ranked here**: the archived Chhavi/Navya `review1-results/manifest.json` lacks the metric-definition ID required by the current validator, and individual LR/GNB fold result JSONs are not present in this clone. The team README explicitly rejects those legacy manifests rather than combining them silently. The saved trip fold IDs and dataset hash do match, but the current metric-definition check cannot certify complete comparability.

## Saved evidence

- `manifest.json`: dataset hash, protocol, metric definition, seed, software versions and fold membership.
- `tree.json`, `forest.json`: per-fold aggregate and per-trip metrics, settings and fit times.
- `tree.joblib`, `forest.joblib`: full-training fitted pipelines; the shared command reloaded and smoke-checked reference-free scoring.
- `aggregate_metrics.csv`, `fold_metrics.csv`: tabular comparison.
- `comparison.json`: output from the shared comparator for these two submissions; absent model submissions remain pending.
- `comparison.png`: simple metric comparison chart.

No hyperparameter search or student-test evaluation was performed; the shared Review 1 command is the untuned baseline protocol.
