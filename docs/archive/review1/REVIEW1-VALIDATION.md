# Review 1 validation and evidence

> Archived Review 1 snapshot. Some status statements and paths describe an earlier repository state; use the root README and current checklists for the final project state.


```powershell
python -m pytest pathsnap_ml/test_pipeline.py -q
python -m pathsnap_ml.review1 train "$HOME/Downloads/Student folder.zip" review1-results
python -m pathsnap_ml.review1 compare review1-results chetha-results navya-results
python -m pathsnap_ml.diagnostics review1-results
```

Each submission directory needs its own manifest and the corresponding model JSON files. Use this revision for submissions: legacy manifests omit the metric-definition identifier and are rejected rather than silently combined. The validator checks exact matching manifests, disjoint fold membership, fold and trip counts, confusion-derived metrics, and reconciliation of Top-1 and confusion counts. It checks consistency, not provenance or whether someone actually ran training. Missing model files remain pending.

The two assigned baselines are evaluated out of fold on the 80 training trips only. Imputation and scaling fit on the training side. Candidates do not use reference routes; accepted labels and candidate misses are evaluation-only on held-out trips. An all-negative classifier can look accurate because there is at most one positive among eight candidates per observation. Top-1 instead asks whether the highest-scoring candidate is the proxy match; candidate misses count as failures, and ties retain candidate order.

Alignment uses ordered reference occurrences with repeat and skip allowed. Distance and path margin reject uncertain labels, but do not independently validate the remaining labels. Free skips, parallel roads, route or GPS errors, and neighboring repeated arcs can still produce inaccurate proxies. Trip splits prevent same-trip leakage, not correlated corridors across trips. Macro F1 uses an explicit binary label set; per-trip all-negative subsets therefore still have two classes in its denominator.

Diagnostics pool only held-out predictions. Distance bins use distance to the nearest generated candidate; speed bins use incoming GPS displacement divided by elapsed time. These are observational strata, not causal explanations. Coverage is computed before excluding uncertain observations; Top-1 is conditional on acceptance. The plot shows pooled confusion, the trip Top-1 distribution, and the condition breakdown. No student-test data or tuning is involved.
