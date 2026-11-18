# Final review and submission checklist

| Requirement and source | Owner | Evidence / implementation | Status and next action | Blocks review? |
| --- | --- | --- | --- | --- |
| Dataset inventory and leakage controls; Dataset Exploration §§3–8 | Chetha | `final-review-evidence/dataset.json`, plots and parser/candidate tests | Verified on training data. Show 208,417 observations, 132,596 accepted, 520 candidate misses | No |
| Five models and grouped comparison; Deliverables §3 | Chhavi and model owners | `review1-integration/comparison.csv`, validation manifests | Verified; confirm with instructor if five distinct families rather than estimators are expected | Conditional |
| Bayesian search and frozen evaluation; Deliverables §4 | Chhavi | `final-review-evidence/outer.json`, trial histories, `student.json`, `model.json` | Verified; no retraining needed | No |
| Saved weights and source; Deliverables §§1–2, p. 2 | Team | `docs/CHHAVI_ML_HANDOFF.md`, model metadata, `deploy/promote_model.py` | Awaiting authorization: publish only exact digest below | Blocks submission |
| HMM/Viterbi comparison; Deliverables §5 and Models and Training §§8–9 | Navya | `pathsnap_ml/viterbi.py`, `tests/test_viterbi.py`, `final-review-evidence/final-routes.json` | Per-observation decoder verified; confirm whether evaluator expects stitched intermediates, which are not implemented | Conditional |
| Folium and representative view; Deliverables §6 | Aryan and team | `ui/map_view.py`, local verified student export | Generation code committed and export works. Capture representative view for PDF; add reference layer offline where available, not in evaluator API | Blocks submission until captured |
| API/UI evidence and two laptops; Deliverables §7 and Models and Training final evaluation | Aryan | `docs/DEPLOYMENT.md`, `final-review-evidence/submission/` | Localhost verified. Run physical checklist below; capture UI, server terminal and actual latency evidence | Blocks final demo sign-off |
| Consolidated report and submission; Deliverables §§1,8 and Assessment Policy §§3–7 | Team | Engineering evidence above and links in `docs/CHHAVI_ML_HANDOFF.md` | Students write and submit the PDF independently; keep evaluator labels out of development | Blocks submission |

## Model approval request

Confirm explicit team permission and distribution eligibility for the exact 126,957-byte `histogram-final-v1` artifact, SHA-256 `9c2acba05bfa1922a412152525dfee624e67ca0845de62c262a424b56d9579a7`. Do not authorize a replacement fit. After approval, branch from latest main, run `python deploy/promote_model.py C:\trusted\model.joblib --dest artifacts/model.joblib`, hash the destination, and stage only that exact binary with `git add -f -- artifacts/model.joblib`. Open a focused PR, verify the retrieved Git blob/file hash in a fresh checkout and confirm `/health` serves it. Do not commit the dataset.

## Aryan's physical rehearsal

1. On laptop 1, use the approved private LAN or hotspot. Follow the Python 3.12 and model promotion commands in [Deployment](DEPLOYMENT.md). Set the trip directory to the ten student trips for rehearsal; keep reference inputs server-side.
2. Start `./deploy/run_api.ps1 -Model artifacts/model.joblib -Trips C:\trips\student_test -Decoder pathsnap_ml.viterbi:decode -Port 8000`. Use the active IPv4 printed by the launcher, not a historical IP. If necessary and authorized, run `open_firewall.ps1 -Port 8000` elevated on a private network. Do not disable the firewall.
3. From laptop 2, call `Invoke-RestMethod http://SERVER_IP:8000/health`; verify the digest, decoder and ten trips. Run `python deploy/smoke_test.py --api http://SERVER_IP:8000 --out physical-smoke.json`, then `./deploy/run_ui.ps1 -Api http://SERVER_IP:8000 -Port 8501`.
4. Select a trip, show proxy scores, latency and Folium layers. Save the physical smoke JSON, UI screenshot and laptop 1 terminal evidence showing the other device's requests. Do not publish raw data or reference geometries. Stop each launcher with Ctrl+C.

Candidate accuracy is binary classification; proxy Top-1 counts accepted observations, including candidate misses as failures. The recorded student comparison was 11,227 versus 11,955 matches out of 13,006 (86.32% vs 91.92%), with directed adjacency 69.59% vs 86.54%. Adjacency is measured over predicted arc changes, not reference agreement or insertion of intermediate arcs. The guidelines ask for one selected road segment per relevant observation plus sequence/continuity evaluation; they do not explicitly mandate full intermediate-arc stitching.
