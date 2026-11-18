# PathSnap ML — engineering handoff

Run from the parent directory of this package with Python 3.12:

```powershell
python -m pip install -r pathsnap_ml/requirements.txt
python -m unittest pathsnap_ml.test_pipeline -v
python -m pathsnap_ml.train "$HOME/Downloads/Student folder.zip" ml-run --trials 8
python -m pathsnap_ml.plot ml-run
```

Use a new output directory each run. Do not publish the raw archive. The runner records its SHA-256, group assignments, fold-level metrics, pipeline settings, trial history, fixed selection, student testing results and fitted pipeline. No evaluator partition is read.

## Review 1: shared untuned protocol

```powershell
python -m pathsnap_ml.review1 train "$HOME/Downloads/Student folder.zip" review1-results
python -m pathsnap_ml.review1 compare review1-results
```

The default runs Chhavi's logistic regression and Gaussian naïve Bayes only. Chetha can append `--models tree forest`; Navya can append `--models histogram`. Exchange the corresponding model JSON files and identical manifest, not raw tracks. The comparator rejects different dataset/split manifests and shows missing models as pending. Keep `review1-v1`, the raw archive hash, feature schema, seed, model settings and fold assignments unchanged for the baseline; coordinate a new protocol revision if the feature or alignment rules change. Per-trip and fold metrics are separate; aggregate figures are equal-weight fold means. Only training trips are read here. The earlier `pathsnap_ml.train` command is a separate post-comparison workflow, not the Review 1 entry point.

The official guidelines call for at least five trained models and exploration across multiple algorithm families. A decision tree and a random forest are distinct estimators, but both are tree-based (histogram boosting is also a tree ensemble). Confirm the instructor's interpretation before treating five named estimators as five distinct families.

## Label and evaluation contract

- Reference alignment uses minimum summed clipped point-to-segment distance over monotone arc **occurrences**, with free repeated occupancy and skips. Forward/backward costs compare complete constrained paths with alternative arc IDs at each observation. Accept a label only within 35 m and with at least an 8 m path-cost margin. This is a weak reference proxy, not verified point-level truth; it can miss detours or mislabel close parallel roads. Coverage is the fraction accepted.
- Candidates use only track, nodes and arcs. The standalone adapter shortlists by sampled segment geometry and midpoints, then ranks exact local-plane point-to-segment distances. It is not an exhaustive nearest-segment search. Coordinates use a trip-local equirectangular approximation. This adapter does not replace Chetha's module; align and retrain with the agreed features/candidate policy if they differ.
- Confident observations with no correct candidate count as Top-1 failures and contribute all-negative classification rows. Uncertain observations are omitted from label-based denominators. Class imbalance is handled at the decision stage by within-observation ranking; models fit the natural candidate prior without reweighting or resampling.
- Selection is mean trip-grouped fold Top-1, then log loss for baseline ties. TPE maximizes the same Top-1 objective. The tuned variant is deployed only if it strictly improves grouped CV Top-1. The reused CV folds are not an unbiased post-search estimate; student test is evaluated after the choices are locked.
- Brier and log loss assess raw binary score quality. No post-hoc calibration is asserted. Thresholded classification uses 0.5. Top-1 ties retain supplied candidate order; feed the same deterministic order in offline and deployed scoring. No route-level claim follows from these metrics.

## Reference-free handoff

Chetha: provide deterministic candidate rows with these features, or coordinate a schema change and retrain. Distance, projected fraction and length are geometric; heading cosine compares directed segment heading with the incoming displacement and is missing for steps below 2 m or invalid elapsed time. Speed uses the preceding track point and seconds elapsed.

```python
from pathsnap_ml.inference import load, predict

artifact = load("ml-run/model.joblib")
rows = [{"trip_id": "demo", "observation_index": 0, "timestamp": 0.0,
         "arc_id": 42, "features": {"distance_m": 5, "heading_cos": None,
                                  "fraction": .2, "length_m": 48, "speed_mps": None}}]
scores = predict(artifact, rows)
```

Navya: output preserves the four identifiers and adds `schema`, `model_version`, and `score`. Scores are independent binary compatibility estimates, not normalized over an observation and not generative emission likelihoods. To use as log potentials, clip away from zero, then take the log. If normalizing, normalize only within the same observation. An observation with no candidates emits no rows; preserve its index in decoder-side track metadata and apply decoder gap policy.

Aryan: load once from a trusted path, use matching dependency versions, and batch-score complete candidate groups. Never load user-supplied joblib files. Keep evaluation labels out of the inference API. Interface is proposed until teammates confirm it.
