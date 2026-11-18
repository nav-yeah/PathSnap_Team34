# PathSnap ML lead handoff

**Owner:** Chhavi Siddarth Wadhwa (PES1UG24CS132). Shared ML protocol, Logistic Regression and Gaussian Naive Bayes, baseline integration, nested histogram tuning, frozen classifier and evaluation evidence are complete. This is an engineering handoff, not submission prose.

## Evidence and decisions

[PR #8](https://github.com/nav-yeah/PathSnap_Team34/pull/8) preserves Aryan's deployment work; [PR #9](https://github.com/nav-yeah/PathSnap_Team34/pull/9) merged verification and [superseded PR #7](https://github.com/nav-yeah/PathSnap_Team34/pull/7)'s unique evidence. See [the validated comparison](../review1-integration/comparison.json) and [final experiment evidence](../final-review-evidence/).

| Untuned model | Candidate accuracy | Macro F1 | Observation Top-1 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 93.39% | 83.99% | 80.34% |
| Gaussian Naive Bayes | 91.38% | 82.94% | 79.03% |
| Decision Tree | 94.74% | 87.71% | 80.67% |
| Random Forest | 94.88% | 87.60% | 81.34% |
| Histogram Gradient Boosting | 94.97% | 88.18% | 81.25% |

- **Untuned validation:** three trip-disjoint folds cover the 80 training trips once; average the fold scores equally. Candidate misses count as Top-1 failures; uncertain aligned labels are excluded. Archive SHA-256 is `4804b0399d1d56f4844b9c74e2f998aae0a281fb5947a1f47ad1bd013214d248`.
- **Selection:** the team picked Histogram Gradient Boosting for comparable Top-1, higher candidate macro F1 and shorter observed fit time than Random Forest. This is not a statistical superiority claim.
- **Nested tuning:** three outer folds; two inner grouped folds; five Optuna TPE trials per outer fold, all 15 completed. Inner groups cover only the outer training trips. The search varied learning rate, maximum leaves and L2. Tuned outer Top-1 was 80.97% versus 81.25% untuned, so the baseline was refit on all training trips and frozen. These outer results compare strategies; student data did not choose hyperparameters.
- **Frozen student evaluation:** on 10 trips, 13,006 of 19,160 observations had accepted proxy labels. Pooled candidate accuracy was 95.98% and macro F1 90.69%. On that same accepted denominator, classifier agreement was 11,227/13,006 (86.32%) versus 11,955/13,006 (91.92%) after Viterbi, +5.60 percentage points. Direct adjacency among predicted arc changes was 69.59% versus 86.54%. These pooled student statistics are not equal-fold training means.
- **Failure example:** trip 00000007 fell from 92.77% to 87.65% despite 100% candidate recall and adjacency rising from 96.93% to 99.18%. This held-out example was not used to tune the decoder. Proxy agreement and direct adjacency are not full-route or independently verified road accuracy. Hidden evaluator results are unavailable.

## Exact artifact and approval

`histogram-final-v1` is a `HistGradientBoostingClassifier` with 70 fitted rounds (scikit-learn 1.9.1), preceded by median imputation with missingness indicators. Feature order in `pathsnap-ml-v1`: distance_m, heading_cos, fraction, length_m, speed_mps. Parameters: learning_rate=0.1, max_iter=70, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0, random_state=34.

**Approval request:** confirm that the team authorizes publishing the exact 126,957-byte artifact from Chhavi's local `outputs/final-histogram/model.joblib` at `artifacts/model.joblib` and that distribution restrictions permit it. SHA-256: `9c2acba05bfa1922a412152525dfee624e67ca0845de62c262a424b56d9579a7`. Official Deliverables p. 2 calls for weights in GitHub. Do not substitute another fit. Until approved, privately transfer only the trusted verified copy and check the hash before loading it.

```powershell
py -3.12 -m pip install -r requirements-dev.txt
py -3.12 -m pytest tests pathsnap_ml/test_pipeline.py -q
Get-FileHash C:\trusted\model.joblib -Algorithm SHA256
py -3.12 deploy/promote_model.py C:\trusted\model.joblib
py -3.12 -m pathsnap_ml.review1 compare C:\evidence\review1-validation tree-review1 review1-integration
```

The first comparison directory must contain the validated Logistic Regression and Naive Bayes results and manifest; regenerate with the shared runner only if these files are missing. See [readiness steps](../FINAL-READINESS.md) for Windows startup and smoke testing. Check `/health` against the digest above.

**Closeout verification:** 42 tests passed, zero failed or skipped; one Starlette TestClient deprecation warning. The current comparator accepted all five records. Promotion copied the exact artifact and checked a reference-free score. A temporary API from merged main reported the expected SHA-256 and decoder and returned success for all ten student trips. See [closeout evidence](../final-review-evidence/closeout/). This was localhost, not a physical two-laptop test.

## Viva reference and remaining owners

- **Why not align by row?** GPS observations and ordered route-arc occurrences are different units; repeat and skip are permitted. The alignment filter excludes uncertain labels and does not independently verify the accepted ones.
- **Candidate accuracy vs Top-1?** Binary candidate accuracy includes many negatives. Top-1 evaluates one best-ranked candidate per accepted observation, including candidate misses.
- **How does decoding use scores?** Clipped log compatibility scores combine with directed transition penalties; they are not calibrated HMM likelihoods. Viterbi can trade local matches for a more connected sequence.
- **Where do references enter?** Prediction uses track, nodes and arcs only. The API reads the reference route for scoring after predictions are final.
- **Navya:** verify decoder behavior and confirm whether full intermediate-arc stitching is required. Current output is per-observation arc selection, not complete stitching.
- **Aryan:** execute and record physical two-laptop rehearsal, including LAN/firewall and client map. Localhost smoke does not establish physical LAN readiness.
- **Chetha:** own any subsequent candidate/schema changes and rerun comparisons if that protocol changes. No preprocessing change is needed for this closeout.
- **Team:** authorize weights, confirm the instructor's interpretation of distinct model families if needed, and independently author the consolidated submission.
