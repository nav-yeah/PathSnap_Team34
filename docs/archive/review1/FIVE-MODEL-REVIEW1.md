# Review 1 integration — engineering decision

> Archived Review 1 snapshot. Some status statements and paths describe an earlier repository state; use the root README and current checklists for the final project state.


**Provisional choice: Random Forest.** It led on the predefined mean-fold Top-1 metric by 0.085 percentage points. Histogram Gradient Boosting is a close alternative: higher candidate accuracy and macro F1, lower Top-1 fold SD, much shorter measured fit time, and comparable inference time. Runtime is indicative, not a controlled benchmark. Both models expose independent compatibility scores; neither score is a calibrated HMM likelihood. Validate downstream decoding before final selection.

The independent histogram verification reproduced Navya's three original Top-1 values exactly and passed reference-free artifact reload. Navya owns the original contribution; the verified submission is separate. The new manifest matches the other four; the original was not edited. The shared validator reconciles trip membership, confusion counts and Top-1 counts, not independent label correctness.

Paired RF-minus-histogram Top-1 across 80 equally weighted trips: +0.053 percentage points, 38 wins / 7 ties / 35 losses. The seeded descriptive trip-bootstrap 95% interval is −0.215 to +0.323 points. The bootstrap does not refit models or account for overlapping fold training sets or shared corridors; do not interpret it as a significance claim.

Review 1 uses accepted reference-proxy labels and counts candidate misses as Top-1 failures. Trip grouping prevents same-trip leakage, not geographic overlap. No tuning, student testing, evaluator data, or final report. The instructions require at least five trained models and exploration of multiple algorithm families; tree, forest, and histogram boosting are all tree-based. Confirm with the instructor whether five distinct families are required.

## Reproduction

```powershell
python -m pytest pathsnap_ml/test_pipeline.py -q
python -m pathsnap_ml.review1 train "$HOME/Downloads/Student folder.zip" baseline --models logistic naive_bayes
python -m pathsnap_ml.review1 train "$HOME/Downloads/Student folder.zip" trees --models tree forest
python -m pathsnap_ml.review1 train "$HOME/Downloads/Student folder.zip" verified --models histogram
python -m pathsnap_ml.review1 compare baseline trees verified
New-Item -ItemType Directory evidence
python five_model_evidence.py evidence baseline trees verified review1-results
```

The last argument points to Navya's original submission. Evidence uses validated fold means and paired per-trip results; binaries and university data are excluded.
