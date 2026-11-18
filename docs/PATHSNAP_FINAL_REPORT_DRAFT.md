# PathSnap: Distributed Map Matching and Route Reconstruction

## Project Report — Draft for Team Review

**Team:** Team 34  
**Course / Institution:** PES University  
**Repository:** https://github.com/nav-yeah/PathSnap_Team34  
**Report status:** Technical draft; physical two-laptop demonstration evidence and final administrative details are pending.

## Team Members

| No. | Student name | SRN | Role / contribution recorded in repository |
|---:|---|---|---|
| 1 | Chetha Samrutha | PES1UG24CS131 | Data and geospatial preprocessing; dataset validation, road candidate generation, reference-proxy alignment and verification |
| 2 | Chhavi Siddarth Wadhwa | PES1UG24CS132 | ML protocol, Logistic Regression and Gaussian Naive Bayes, model comparison, nested tuning and frozen HGB model |
| 3 | **[Confirm full name]** | **[Confirm SRN]** | API, Streamlit UI and deployment — repository assigns this work to Aryan |
| 4 | **[Confirm full name]** | **[Confirm SRN]** | HMM / Viterbi sequential decoder — repository handoffs assign this work to Navya |

> Complete the two placeholder rows and verify all four names, SRNs, role descriptions and order with the team before submission. The code repository gives first names for Aryan and Navya but does not provide their full names or SRNs in the files reviewed.

---

## Abstract

PathSnap is a distributed map-matching and route-reconstruction system that maps noisy GPS observations to likely directed road segments and selects a plausible sequence across a trip. For each GPS observation, the system generates a bounded set of nearby road-arc candidates and represents each observation–candidate pair using geometric and movement-related features. A Histogram Gradient Boosting (HGB) classifier assigns an independent compatibility score to each candidate. A Viterbi sequence decoder then combines clipped log classifier scores with directed-network transition constraints and movement plausibility to select one road arc per observation.

Five classifiers were compared using trip-grouped validation. HGB was selected because it achieved the highest candidate-level Macro F1 among the compared models, a competitive observation-level Top-1 result, and shorter measured fitting time than Random Forest. Nested grouped hyperparameter tuning did not improve the baseline: the tuned outer-fold Top-1 was 80.97%, compared with 81.25% for the untuned baseline. The baseline HGB model was therefore retained and frozen.

On ten student-test trips, 13,006 observations received accepted reference-route proxy labels. HGB classifier-only predictions matched 11,227 of these labels (86.32%); HGB with Viterbi decoding matched 11,955 (91.92%), an increase of 5.60 percentage points. Direct adjacency among predicted arc changes increased from 69.59% to 86.54%. These are agreement results against alignment-derived proxy labels, not independently verified observation-level physical ground truth. A physical two-laptop demo and its screenshots/logs remain pending and must be added before the report is finalised.

## 1. Introduction and Problem Statement

GPS receivers produce observations that may be noisy, irregularly spaced or displaced from the road on which a vehicle is travelling. A single GPS coordinate can be close to several road segments, particularly at intersections or on parallel roads. Selecting the nearest road independently at every observation can result in implausible jumps, wrong turns or a disconnected sequence.

PathSnap addresses this by combining candidate generation, supervised candidate scoring and sequential decoding. It also separates the inference pipeline from reference-route evaluation: the inference path uses GPS tracks and road-network data, while reference routes are accessed only after predictions have been produced for aggregate evaluation.

### Objectives

1. Generate plausible directed road-segment candidates for every GPS observation.
2. Compute geometric and movement-related features for observation–candidate pairs.
3. Compare multiple classifiers and select a reproducible model using trip-grouped validation.
4. Decode a more coherent sequence using candidate scores and directed-road-network transition compatibility.
5. Expose the pipeline through an API and map-based interface.
6. Evaluate candidate ranking, proxy-label agreement and route continuity while documenting limitations.

## 2. Dataset and Reference Alignment

Each trip is represented by four headerless, whitespace-separated files:

| File | Contents | Interpretation |
|---|---|---|
| `.track` | Longitude, latitude and timestamp in seconds | Ordered GPS observations |
| `.nodes` | Longitude and latitude | Road-network node coordinates; row index is node ID |
| `.arcs` | Directed from-node and to-node IDs | Road segments; row index is arc ID |
| `.route` | Ordered arc IDs | Reference route sequence |

The project’s training audit covered 80 trips and 208,417 GPS observations. The files do not provide a direct one-to-one mapping between GPS observations and reference arcs: the number of observations does not have to equal the number of route arcs.

To create supervised labels, the project aligns GPS observations to occurrences in the ordered reference route. The alignment permits repeated occupancy of an arc and skips over route occurrences. A label is accepted only when the selected reference-route segment is within 35 metres and the best complete-path alternative with a different arc ID costs at least 8 metres more. Observations that do not meet these confidence rules are treated as uncertain and excluded from label-based metrics. Accepted labels whose arc is absent from the generated candidate set count as Top-1 failures.

The training audit accepted 132,596 proxy labels, corresponding to 63.6205% coverage. There were 520 candidate misses among accepted labels, giving candidate recall of 99.6078%. These values describe the project’s alignment and candidate coverage rules; they do not establish that the labels are independently verified physical ground truth.

## 3. Candidate Generation and Feature Engineering

Candidate generation uses the GPS track and road-network geometry; it does not use the reference route. Road arcs are sampled along their geometry at approximately 20-metre intervals or less. A spatial index retrieves nearby samples, the shortlist is augmented with nearby arc midpoints, and exact point-to-segment distances are calculated within the bounded shortlist. Up to eight closest directed arcs are retained, preserving their original arc IDs.

The project uses a trip-local equirectangular coordinate approximation. The median GPS coordinate is used as the local origin; longitude scaling includes a cosine-of-origin-latitude factor. This makes trip-scale distance calculations practical but is not a global geodesic projection.

Each observation–candidate pair contains the following five features:

| Feature | Description |
|---|---|
| `distance_m` | Distance from the GPS observation to the candidate road segment, in metres |
| `heading_cos` | Cosine similarity between directed road-segment direction and incoming GPS movement |
| `fraction` | Clamped projected position along the candidate segment, from 0 to 1 |
| `length_m` | Candidate segment length, in metres |
| `speed_mps` | GPS displacement divided by actual elapsed time, in metres per second |

Missing heading and speed values are represented as missing values and handled by the fitted preprocessing pipeline using median imputation and missingness indicators. Logistic Regression and Gaussian Naive Bayes also use feature scaling; the tree-based models do not require the same scaling step.

A sampled exact-nearest check compared 480 observations (six evenly spaced observations per training trip) against an exhaustive scan of each trip’s arcs. All 480 sampled shortlists retained an exact nearest segment. This is sampled evidence and is not a guarantee that every global nearest segment will always appear in the bounded shortlist.

## 4. Model Comparison

Five classifiers were evaluated using three-fold grouped cross-validation by trip. All observations from a trip remain within one fold to reduce leakage between training and validation observations from the same trip. The table reports equal-weight means across folds and uses accepted reference-route proxy labels.

| Model | Candidate accuracy | Macro F1 | Observation Top-1 |
|---|---:|---:|---:|
| Logistic Regression | 93.39% | 83.99% | 80.34% |
| Gaussian Naive Bayes | 91.38% | 82.94% | 79.03% |
| Decision Tree | 94.74% | 87.71% | 80.67% |
| Random Forest | 94.88% | 87.60% | 81.34% |
| Histogram Gradient Boosting | 94.97% | 88.18% | 81.25% |

Candidate accuracy is calculated over candidate rows and can be high because most candidate rows are negative examples. Observation Top-1 evaluates whether the highest-scoring candidate matches the accepted proxy label for each observation; candidate-generation misses count as failures. Candidate recall is a separate measure of whether the proxy-labelled road was included among the generated candidates.

HGB was selected because it achieved the highest candidate Macro F1, had a competitive observation Top-1 result, and had shorter measured fitting time than Random Forest. The small Top-1 difference between HGB and Random Forest is not evidence of statistical superiority.

## 5. Histogram Gradient Boosting and Hyperparameter Tuning

Histogram Gradient Boosting is an ensemble method that builds decision trees sequentially. Each new tree contributes to correcting errors made by the existing ensemble. The histogram-based implementation groups numerical feature values into bins when considering candidate tree splits, improving training efficiency.

In PathSnap, HGB produces independent compatibility scores for observation–candidate pairs. The five features are imputed using training-fold medians with missingness indicators. The frozen baseline settings recorded in the repository are:

- Learning rate: 0.1
- Maximum iterations: 70
- Maximum leaf nodes: 31
- Minimum samples per leaf: 20
- L2 regularization: 0
- Random seed: 34

Nested grouped tuning used three outer folds, with two inner grouped folds within each outer training split. Optuna’s TPE sampler ran five trials per outer fold, for 15 trials in total. The search varied learning rate, maximum leaf nodes and L2 regularization. The tuned approach achieved 80.97% outer-fold Top-1, compared with 81.25% for the untuned baseline. Since tuning did not improve the selected objective, the team retained and froze the baseline model.

The frozen artifact is identified in the engineering handoff by SHA-256:
`9c2acba05bfa1922a412152525dfee624e67ca0845de62c262a424b56d9579a7`.

**Submission condition:** the repository checklist records publication of the model weights as awaiting explicit approval and distribution eligibility. Confirm that approval and the actual artifact availability before claiming the weights are included in the submitted repository.

## 6. Sequential Decoding: HMM Framing and Viterbi

The classifier makes a local prediction for each GPS observation. Sequential decoding uses candidate compatibility and directed-network movement constraints to select a sequence across the trip.

- **Observation:** GPS location and timestamp.
- **Hidden state:** candidate directed road arc occupied at that observation.
- **Emission compatibility:** the HGB classifier’s score for an observation–arc candidate.
- **Transition compatibility:** whether travel from the previous candidate to the next is plausible on the directed road network, given elapsed time and GPS displacement.

The repository’s decoder clips classifier scores away from zero and takes their logarithms to form compatibility potentials. It then uses dynamic programming: for every candidate at the current observation, it considers predecessor candidates, adds the transition penalty and retains the best predecessor. A traceback over the stored predecessor choices produces the selected sequence.

**Important modelling qualification:** these classifier scores are independent compatibility estimates. They are not calibrated generative HMM emission likelihoods. Therefore, describe the implementation as HMM-style sequence modelling / Viterbi decoding with classifier-derived log compatibility scores, rather than claiming that the classifier outputs are calibrated HMM probabilities.

The decoder checks directed connectivity and movement plausibility. Long or invalid time gaps and impossible transitions can cause a new sequence segment to begin. It returns one selected arc per observation; it does not insert every intermediate road arc to produce a fully stitched path. This distinction should be retained when describing route reconstruction and continuity.

## 7. Student-Test Results

On ten student-test trips, 19,160 observations were evaluated and 13,006 received accepted proxy labels.

| Method | Proxy Top-1 agreement | Correct accepted observations | Direct adjacency |
|---|---:|---:|---:|
| HGB classifier only | 86.32% | 11,227 / 13,006 | 69.59% |
| HGB + Viterbi | 91.92% | 11,955 / 13,006 | 86.54% |

Viterbi increased pooled proxy Top-1 agreement by 5.60 percentage points and improved direct adjacency. Results varied by trip: for trip 00000007, proxy Top-1 decreased from 92.77% to 87.65%, even though direct adjacency improved. Better continuity therefore does not guarantee higher proxy-label agreement for every trip.

These figures are from the student-test partition, not the separate evaluator dataset. Evaluator-set accuracy is not available and must not be inferred from the student-test results.

## 8. System Architecture and Implementation

PathSnap consists of a FastAPI backend, an HGB model artifact, a configurable decoder, a Streamlit interface and a Folium map.

### Laptop 1 — API and inference

The FastAPI service loads the model and trip inputs, generates candidate road arcs, computes features, scores candidates and applies the configured decoder. It exposes:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model state, version and fingerprint, decoder and available-trip count |
| `GET /trips` | Available trip IDs and summary information |
| `POST /predict` | Runs prediction and returns per-observation predicted arcs, route summary, continuity, timing and map geometry |
| `GET /docs` | Interactive API documentation |

Inference reads `.track`, `.nodes` and `.arcs`. The reference `.route` is accessed only after predictions are final for server-side aggregate scoring. Reference arc IDs and per-observation correctness details are not exposed to the client API.

### Laptop 2 — client interface

The Streamlit interface communicates with the FastAPI server, lists trips and displays predicted route information and map layers using Folium. The UI does not contain the model or university trip dataset; it sends a trip identifier to the API and displays the response.

### Physical deployment evidence — pending

The repository’s submission checklist records the physical two-laptop rehearsal as outstanding. The final report must include actual evidence from two separate devices, not a localhost-only substitute:

1. UI screenshot showing the selected trip and reconstructed route map.
2. `/health` response showing loaded model/version, expected SHA-256, active decoder and trip count.
3. Ten-trip client smoke-test output and its JSON file.
4. Laptop 1 server logs showing requests from Laptop 2 and measured request latency.
5. Unknown-trip 404 check.
6. A representative exported map or report-ready map image.

Add the real evidence files, test date, laptop/network setup and observed values here after the physical run. Do not invent screenshots, logs, latency numbers or successful network checks.

## 9. Testing and Verification

The repository records automated tests covering candidate generation and feature schema, model compatibility, API endpoints and error paths, reference-data isolation, decoder behaviour, UI integration and map rendering. The preprocessing verification document records a run of 44 passing tests with one Starlette deprecation warning. Other later closeout records report different test commands/scopes; the final report should identify the exact command and date associated with each reported test count rather than combining them into a single unexplained number.

The team’s localhost API smoke test is recorded as completed. It does not establish a physical two-device LAN demonstration. The physical smoke-test output and server logs must be appended after the separate-laptop run.

## 10. Limitations

1. Reference-route labels are alignment-derived proxies and can be wrong or ambiguous; they are not independently verified GPS ground truth.
2. Only 63.6205% of the training GPS observations received accepted proxy labels under the documented confidence rules.
3. Candidate generation uses a bounded spatial shortlist. The 480-point exact-nearest audit is sampled evidence, not an exhaustive guarantee.
4. Trip-grouped validation reduces same-trip leakage but cannot remove all correlation between trips that share roads or corridors.
5. HGB candidate scores are independent compatibility estimates, not calibrated generative likelihoods.
6. Improved route continuity can coincide with lower proxy agreement on an individual trip.
7. The decoder emits one road arc per observation and does not insert all intermediate road arcs to make a fully stitched network path.
8. Evaluator-set performance is unavailable; student-test metrics must not be represented as evaluator accuracy.
9. Timestamp epoch and timezone are undocumented; the system uses differences between timestamps as elapsed time.
10. The local coordinate projection is an approximation, and the road-candidate shortlist is not an exhaustive global nearest-segment search.

## 11. Conclusion

PathSnap combines GPS-to-road candidate generation, HGB candidate scoring and Viterbi sequence decoding to map noisy GPS observations to plausible directed road arcs. Trip-grouped validation supported selecting the frozen HGB baseline, while nested tuning did not improve the baseline Top-1 objective. On ten student-test trips, Viterbi improved pooled proxy-label Top-1 agreement and direct adjacency compared with classifier-only predictions, although some trip-level proxy agreement decreased. All accuracy figures must be interpreted as agreement with alignment-derived proxy labels rather than independently verified physical ground truth.

The technical implementation and engineering evidence are present in the repository, but the consolidated submission still requires final team identity verification, explicit model-artifact approval where applicable, representative map evidence, the physical two-laptop demonstration record, final references and formatting, and confirmation of any instructor requirement for intermediate-arc stitching.

## References

1. PES University, *PathSnap: Dataset Description and Exploration*, supporting project document, sections 3 and 5–8.
2. Scikit-learn documentation, `HistGradientBoostingClassifier` (add the exact versioned documentation URL consulted by the team).
3. Optuna documentation, Tree-structured Parzen Estimator (TPE) sampler (add exact versioned reference).
4. FastAPI documentation (add URL and access date).
5. Streamlit documentation (add URL and access date).
6. Folium documentation (add URL and access date).
7. The project repository: https://github.com/nav-yeah/PathSnap_Team34 (access date: [insert date]).

## Appendices to attach before submission

- A. Dataset inventory and archive fingerprint (do not attach restricted raw university data).
- B. Preprocessing and candidate-generation evidence, including Figures 1–2 and per-trip audit summaries.
- C. Five-model comparison, grouped-fold metrics, nested tuning trial history and frozen model metadata.
- D. Decoder tests and per-trip classifier-versus-decoder results.
- E. API health response, smoke-test JSON, server logs and unknown-trip 404 check.
- F. Physical two-laptop UI screenshot and representative Folium map.
- G. Reproduction commands, dependency versions and model SHA-256 verification.
