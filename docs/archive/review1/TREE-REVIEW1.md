# Decision Tree and Random Forest — Review 1 evidence

> Archived Review 1 snapshot. Some status statements and paths describe an earlier repository state; use the root README and current checklists for the final project state.


Untuned, three-fold grouped cross-validation on all 80 training trips. Equal-weight fold means:

| Model | Candidate accuracy | Macro F1 | Observation Top-1 |
| --- | ---: | ---: | ---: |
| Decision Tree | 94.74% | 87.71% | 80.67% |
| Random Forest | 94.88% | 87.60% | 81.34% |

Labels are accepted reference proxies. Candidate misses count as Top-1 failures. No student-test or evaluator data was used. Both refitted artifacts passed reference-free reload checks. The existing five ML tests passed.

```powershell
python -m pathsnap_ml.review1 train "$HOME/Downloads/Student folder.zip" tree-review1-generated --models tree forest
python -m pathsnap_ml.review1 compare tree-review1-generated
```

The checked-in submission is in `tree-review1/`. Compare with other submissions by passing their directories to the same command. Legacy manifests missing the metric-definition field must be regenerated, not edited. The dataset and model binaries are not committed.
