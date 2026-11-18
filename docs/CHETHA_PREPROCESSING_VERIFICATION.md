# Chetha Preprocessing and Dataset Verification

**Scope.** This document records an audit of the merged PathSnap implementation and a fresh scan of the permitted training partition. It does not redesign preprocessing, retrain a model, or certify `.route` labels as physical ground truth. Dataset counts below are reproduced from the local `Student folder.zip` archive; the audit output records its SHA-256 so the source version is identifiable.

## 1. Input format and parsing

The supplied PES Dataset Description and Exploration document defines the raw files as headerless, whitespace-separated tables:

| File | Columns | Parsed dtype and shape | Meaning |
|---|---|---|---|
| `.track` | longitude, latitude, timestamp seconds | float64, `(observations, 3)` | Ordered GPS fixes |
| `.nodes` | longitude, latitude | float64, `(nodes, 2)` | Network node coordinates; row index is node ID |
| `.arcs` | from-node ID, to-node ID | int64, `(arcs, 2)` | Directed segment endpoints; row index is arc ID |
| `.route` | arc ID | int64, one-dimensional sequence | Chronological reference-arc sequence |

The documentation says track sampling is nominally 1 Hz and that about 97% of intervals are one second. It specifies timestamps in seconds but does not specify a Unix epoch or timezone. The implementation uses differences between consecutive timestamps as elapsed seconds; it does not convert timestamps to calendar time.

The fresh scan read all 80 training trips: 80 files of each type, with the expected column counts and dtypes above. All training rows had finite coordinates within longitude/latitude bounds; all arc endpoints and route IDs were in range. No malformed trip was found. There were no duplicate full `.track` rows and no nonpositive time intervals. The loader/audit use `numpy.loadtxt`; malformed numeric input or inconsistent columns raises during parsing or shape validation. The audit checks data validity, while the inference trip store separately rejects empty tables, invalid shapes, nonfinite coordinates, out-of-range coordinates and invalid arc endpoints. Duplicate network rows are not globally rejected: arc row identity is meaningful and parallel arcs must remain distinct. These statements describe the checked archive and existing code, not a general guarantee for arbitrary future files.

Timestamp results: 201,053 of 208,337 intervals (96.50%) were exactly 1.0 seconds within 1 microsecond. No interval was nonpositive. The largest gap was 153,064 seconds. This agrees with nominal 1 Hz sampling but shows that long gaps exist; features use the actual elapsed interval rather than assuming one second.

## 2. Coordinate projection

`pathsnap_ml.data.candidates()` reads coordinates in `[longitude, latitude]` order and chooses the component-wise median GPS coordinate as the trip-local origin. It converts degree differences to local metres with a constant of 111,195 metres per degree and multiplies the east/longitude scale by `cos(origin_latitude)`. North/latitude uses the same metres-per-degree constant without the cosine factor. Candidate generation, point-to-segment distances, road lengths and the exact-nearest audit all use this same local plane.

This is an equirectangular local approximation, suitable for these trip-scale calculations, not a global geodesic projection. Small scale error can grow over large extents or at high latitudes. Coordinates and axes were visually inspected in the regenerated track overlay.

## 3. Road geometry

`geometry()` projects each point onto a segment using the dot product divided by squared segment length. It clamps the fraction to `[0, 1]`, then computes Euclidean residual distance from the clamped projection. Segment length is the Euclidean norm of endpoint difference. Since inputs are in the local metre plane, outputs are metres.

The denominator is floored at `1e-12` to avoid division by zero. A zero-length segment therefore projects to its endpoint (fraction zero), has its endpoint distance measured, and has length zero. The training audit found 100 zero-length arc geometries across all trips; they were preserved rather than silently removed. Existing pipeline tests cover geometry boundary/degenerate behavior. No reproducible geometry defect was found, so this implementation was left unchanged.

## 4. Candidate generation

`candidates(track, nodes, arcs)` does not accept or read `.route`. Each directed arc is sampled along its segment at spacing no greater than approximately 20 m, including endpoints. A `cKDTree` returns up to 64 nearby samples for each GPS point. The shortlist is augmented with up to eight arcs near the point by arc-midpoint distance. The implementation unions the resulting arc IDs, computes exact clamped point-to-segment distances for that bounded pool, and retains the closest eight candidates.

`np.union1d` sorts unique arc IDs, then `np.argsort(..., kind="stable")` ranks distances. Thus equal final distances retain ascending arc-ID order within the shortlist. The spatial index determines shortlist membership, so this is not an exhaustive all-arcs guarantee; an exact nearest arc can in principle be excluded before final ranking. The audit's check is sampled evidence, not proof for all observations.

The candidate generator emits eight slots for the supplied dataset and keeps original directed arc row IDs. It uses only GPS, nodes and arcs. The deterministic audit sampled six evenly spaced observations per training trip (480 total) and compared each shortlist's minimum distance with an exhaustive scan over every arc; ties within `1e-8` m counted as equal. All 480 sampled shortlists retained an exact nearest segment. This result must be described as “480 sampled checks passed,” not “the shortlist is always exact.”

## 5. Ordered reference-proxy alignment

`align()` is separate from candidate creation. The `.route` sequence is used for evaluation/reference-label construction; inference loading reads only `.track`, `.nodes` and `.arcs`. Alignment builds point-to-segment distances to route occurrences in order. Dynamic programming uses cumulative minima, allowing an observation to remain on a route occurrence or advance to a later occurrence, including skipped route occurrences. Repeated arc IDs remain separate positions in the ordered route arrays during path search. The returned class is the original arc ID, so identical arc IDs at different route positions are the same classification target.

The path cost clips point distance at 100 m. A proxy label is accepted only if its point-to-route-segment distance is at most 35 m and the best full-path alternative with a different arc ID is at least 8 m more costly. Otherwise the observation is set to `-1` (uncertain/rejected). Repeated occurrences with the same arc ID are not treated as a different-ID ambiguity alternative; occurrence identity affects ordering, while the classification target remains arc ID. Candidate misses on accepted proxy labels remain failures for observation-level Top-1 and are counted against candidate recall.

These are **reference-route proxy labels**, not independently verified physical truth. The official dataset document explicitly says supplied routes may be limited, unverified or erroneous and GPS fixes may not lie on the road exactly. Close parallel roads, route errors and GPS noise can still yield a confident but incorrect proxy. The current protocol was not changed.

## 6. Shared feature definitions

Training's `FEATURES` tuple and inference's schema check/matrix conversion use exactly this order:

1. `distance_m` — point-to-candidate-segment distance in metres.
2. `heading_cos` — cosine between directed segment vector and incoming GPS displacement; circular heading is represented without a discontinuity at 0/360 degrees.
3. `fraction` — clamped projection position along the candidate segment, from 0 to 1.
4. `length_m` — candidate segment length in metres.
5. `speed_mps` — incoming GPS displacement divided by actual elapsed seconds.

Heading is missing when displacement is under 2 m or elapsed time is nonpositive. Speed is missing for the first fix and when elapsed time is nonpositive. Nonfinite values and `None` become `NaN` at inference. The model pipeline fits median imputation plus missingness indicators on each training fold; inference uses the saved pipeline. No scaling is added for the tree models or frozen histogram model. Identifier fields and route IDs are not in the feature vector.

## 7. Reproduced training statistics

The full audit ran on the training partition only. Student-test files were not opened for this recomputation; evaluator data was not used.

| Measurement | Reproduced result |
|---|---:|
| Training trips | 80 |
| GPS observations | 208,417 |
| Accepted reference-proxy labels | 132,596 |
| Accepted-label coverage | 132,596 / 208,417 = 63.6205% (63.62%) |
| Candidate misses among accepted labels | 520 |
| Candidate recall | (132,596 - 520) / 132,596 = 99.6078% (99.61%) |
| Exact-nearest shortlist sample | 480 / 480 retained; 0 sample misses |
| Nonpositive timestamp intervals | 0 |
| Duplicate full GPS rows | 0 |
| Zero-length arc geometries | 100 |
| Incoming speed estimates above 55 m/s | 980 |

Candidate recall is a coverage metric, not classifier accuracy. Sampled exact-nearest retention is not an exhaustive guarantee. Coverage describes the shared acceptance rule, not verified label accuracy.

The high-speed count is descriptive only. The candidate pipeline does not automatically remove those observations; the thresholds are not asserted as a validated vehicle-speed limit.

## 8. Figures and tests

I reran the existing `audit_geometry.py` on the training archive into `final-review-evidence/chetha-preprocessing-verification/`. Its `distributions.png` and `track-overlay.png` are byte-identical to the committed figures in `final-review-evidence/`; no figure was replaced. Their content is generated from nearest-candidate distance, speed, heading cosine, accepted/excluded proxy counts, per-trip coverage, candidate recall and the first training trip's local network overlay. The regenerated files and hashes are in the audit evidence folder.

The full project suite was run after installing the repository's pinned test dependencies from `requirements-dev.txt`:

```text
py -m pytest -q
44 passed, 1 warning in 8.78s
```

The warning is a Starlette deprecation notice for `httpx` integration in the API test client. No test failed. No implementation fix was required.

## 9. Limitations and human confirmation

- The route file is treated as chronological reference information per the dataset description, but the labels remain proxies and the source warns they may be imperfect.
- Timestamp seconds have no documented epoch/timezone. The pipeline appropriately uses elapsed differences, but calendar-time interpretation is unsupported.
- The local equirectangular plane is an approximation.
- Candidate spatial indexing is a bounded shortlist; 480 exhaustive checks are evidence on a sample only.
- The 100 degenerate arcs are retained. Their distance calculation is defined, but the source network's intended semantics for such arcs are unknown.
- `TripStore` validates parser shape and coordinate/arc ranges but does not explicitly enforce increasing timestamps. The verified training archive has zero nonpositive intervals; I did not modify Aryan-owned API code within this audit scope.
- Duplicate observations are counted by the audit, not automatically removed. None occurred in the training archive.
- Team members should confirm acceptance thresholds and describe all resulting labels as proxy labels in the final report.

## Reproduction

From the repository root:

```powershell
py audit_geometry.py "$HOME\Downloads\Student folder.zip" final-review-evidence/chetha-preprocessing-verification
py -m pytest -q
```

The audit output directory must not already exist; the script intentionally creates a new output directory rather than overwriting prior evidence. It writes per-trip `dataset.json`, `distributions.png`, and `track-overlay.png`. `source_file_verification.json` and `verification_summary.json` in the same folder record the independent table scan and exact denominators used here.
