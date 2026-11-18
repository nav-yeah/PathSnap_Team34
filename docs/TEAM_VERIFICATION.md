# Final technical sign-off record

Engineering record only; this does not record teammate approvals or constitute the academic report.

| Requirement / official source | Owner | Status | Evidence and next action |
| --- | --- | --- | --- |
| Input inventory and preprocessing; Dataset Exploration §§3–8 | Chetha | COMPLETE — AWAITING REVIEW | `final-review-evidence/dataset.json`, `audit_geometry.py`, and shared data tests. Confirm the existing feature and candidate policy; no schema changes. |
| Five models, group-aware evaluation and Bayesian optimisation; Deliverables §§3–4 | Chhavi / model owners | COMPLETE | `review1-integration/`, `final-review-evidence/outer.json`, trial histories and `docs/CHHAVI_ML_HANDOFF.md`. No rerun or retraining necessary. |
| Saved model weights; Deliverables p. 2 | Team | AWAITING EXTERNAL APPROVAL | Authorize the exact digest below for publication at `artifacts/model.joblib`, subject to distribution restrictions. See `docs/SUBMISSION-CHECKLIST.md`. |
| Sequential inference and comparison; Deliverables §5 and Models and Training §§8–9 | Navya | COMPLETE — AWAITING REVIEW | `pathsnap_ml/viterbi.py`, `evaluate_routes.py`, `final-review-evidence/final-routes.json`, and decoder tests. Confirm whether evaluator expects intermediate arc stitching. |
| API/UI, latency and reference isolation; Deliverables §7 | Aryan | COMPLETE — AWAITING REVIEW | Local suite and ten-trip HTTP smoke in `final-review-evidence/team-verification/`. Use actual model/decoder from health. |
| Two distinct laptops; Models and Training final evaluation | Aryan | BLOCKED — PHYSICAL HARDWARE REQUIRED | Run the two-device checklist below and retain client smoke and server logs. |
| Representative Folium view; Deliverables §6 | Aryan / team | INCOMPLETE | Exporter code is committed and local HTML export verified. Capture representative PDF view and add reference layer offline where applicable. Keep reference geometry out of reviewer API. |
| Consolidated PDF and submission integrity; Deliverables §§1, 8; Assessment Policy §§3–7 | Whole team | INCOMPLETE | Students independently write the report and verify its evidence and submission; no hidden evaluator data used here. |

## Chetha — preprocessing review

Training uses 80 trips and 208,417 observations, with 132,596 accepted proxy labels (63.62%) and 520 candidate misses on accepted points. The deterministic audit sampled six points per trip; all 480 shortlists retained an exact nearest segment (ties allowed; not proof for all points). Track columns are lon, lat, seconds; the local plane scales longitude by cosine of median trip latitude. Candidate generation ranks clamped point-to-segment distances with eight slots when available and never reads a reference route. The ordered reference alignment preserves repeated arc occurrences, allows skips, and rejects distant or ambiguous assignments. Median imputation handles missing heading/speed; inference and training share distance, heading cosine, fraction, length and speed.

## Navya — decoder review

Valid scores become clipped log compatibility potentials, not normalized emission likelihoods. Each step adds the best predecessor score and movement penalty, recording the predecessor for deterministic traceback. Road transitions use shared directed endpoints or at most one directed bridge arc, and apply elapsed-time/displacement checks. A disconnected transition cannot continue a chain; the decoder can start a new chain, so successive emitted states may still be disconnected. New tests verify directed bridge acceptance, reversed bridge rejection and single-point invalid IDs with zero scores. Direct adjacency counts shared endpoints across collapsed arc changes, not bridge reachability or reference agreement. The emitted sequence is not fully stitched; the official documents require per-observation matching and sequence/continuity evidence but do not explicitly demand all intermediate arcs.

The frozen comparison uses 13,006 accepted student observations for both arms: 11,227 classifier and 11,955 decoder matches (86.32% vs 91.92%); directed adjacency is 69.59% vs 86.54%. On trip 00000007 classifier agreement was 92.77% versus 87.65% decoded despite adjacency improving from 96.93% to 99.18%. This is a held-out trade-off, not a target for tuning.

## Aryan — final demonstration

1. On laptop 1 verify the trusted `histogram-final-v1` artifact (126,957 bytes, scikit-learn 1.9.1) has SHA-256 `9c2acba05bfa1922a412152525dfee624e67ca0845de62c262a424b56d9579a7`. Promote with `python deploy/promote_model.py C:\trusted\model.joblib`.
2. On a permitted private network run `./deploy/run_api.ps1 -Model artifacts/model.joblib -Trips C:\trips\student_test -Decoder pathsnap_ml.viterbi:decode -Port 8000`. Use its active IPv4. If needed, use the existing narrow firewall rule, not a firewall shutdown.
3. On laptop 2 run `Invoke-RestMethod http://SERVER_IP:8000/health`, check hash and decoder, then `python deploy/smoke_test.py --api http://SERVER_IP:8000 --out physical-smoke.json` and `./deploy/run_ui.ps1 -Api http://SERVER_IP:8000 -Port 8501`.
4. Select a trip and show actual classifier/decoder proxy scores, continuity, latency and map layers. Capture UI, server terminal, model fingerprint, client smoke output and time from two separate laptops. Keep the recorded localhost run as a fallback, labeled as such. Use Ctrl+C to shut down each launcher.
5. Run `python -m ui.map_view --api http://SERVER_IP:8000 --trip 00000052 --out map.html` for a representative map. The report view is a separate student task. Do not distribute raw university data or local data-bearing HTML without permission.
