# Final review handoff

## Frozen model

Use the exact `final-histogram/model.joblib` from the local run, not a new fit:

- SHA-256: `9c2acba05bfa1922a412152525dfee624e67ca0845de62c262a424b56d9579a7`
- 126,957 bytes; `histogram-final-v1`; `HistGradientBoostingClassifier`; 70 fitted rounds; scikit-learn 1.9.1.
- Schema `pathsnap-ml-v1`: distance_m, heading_cos, fraction, length_m, speed_mps.
- Learning rate 0.1, maximum 70 iterations, 31 leaves, minimum 20 samples per leaf, L2 0, seed 34. See `final-review-evidence/model.json` and `protocol.json` for provenance.

Deliverables p. 2 calls for saved weights in GitHub. Obtain explicit approval to publish these exact weights before tracking them; until then transfer the trusted artifact privately and verify its hash. Never load an untrusted joblib file.

## Readiness checklist

| Source | Deliverable and evidence | Status / owner |
| --- | --- | --- |
| Dataset Exploration §§3–8 | `audit_geometry.py`, dataset inventory and plots in `final-review-evidence/`; candidate and alignment tests | Verified on training archive; Chetha owns preprocessing |
| Models and Training §§2–7; Deliverables §§3–4 | `review1-integration/`, nested TPE trials, outer metrics and frozen student metrics | Verified; baseline histogram retained after tuning |
| Deliverables p. 2 | Weights, source, configuration | Verified locally; publication authorization pending |
| Models and Training §§8–9; Deliverables §5 | `pathsnap_ml/viterbi.py`, decoder tests and `final-routes.json` | Per-observation decoding verified; Navya review pending |
| Deliverables §§6–7 | Folium map generation via `ui.map_view`, API/UI tests and localhost smoke | Local verification only; team must capture representative report view and execute two-laptop rehearsal |
| Assessment and Plagiarism Policy; Deliverables §§1, 8 | Genuine evidence, isolated evaluator data, student-authored consolidated PDF | Team writes and submits the report |

The model guidelines explicitly ask for a predicted road segment for each relevant observation, plus sequence and continuity evaluation. They do not explicitly require inserting every intermediate arc. This implementation does not stitch them: a bounded one-bridge transition is not full path reconstruction. Confirm evaluator expectations with Navya; do not present selected arcs or proxy agreement as complete-route accuracy.

## Student trip 00000007

The frozen run recorded 1,104/1,190 classifier matches (92.77%) and 1,043/1,190 decoded matches (87.65%); 164 of 1,354 observations were excluded. Candidate recall on accepted observations was 100%. Both had zero unmatched. Direct adjacency rose from 253/261 to 121/122, while accepted matches fell by 61 net. This shows a trade-off between proxy agreement and continuity; it does not isolate which emission, heading, or transition caused each disagreement. No decoder changes were selected using this trip.

## Aryan's Windows rehearsal

Install `requirements-dev.txt` in an activated Python 3.12 environment. On laptop 1, from the repository root:

```powershell
Get-FileHash C:\trusted\model.joblib -Algorithm SHA256
python deploy/promote_model.py C:\trusted\model.joblib
.\deploy\run_api.ps1 -Model artifacts\model.joblib -Trips C:\trips\student_test -Decoder pathsnap_ml.viterbi:decode -Port 8000
```

Pick the active IPv4 address printed by the launcher. On a trusted private LAN, if needed, an administrator may run `.\deploy\open_firewall.ps1 -Port 8000`. Never turn off the firewall. Keep reference files on laptop 1 only.

On laptop 2, replace `SERVER_IP` with the address above:

```powershell
Invoke-RestMethod http://SERVER_IP:8000/health
python deploy/smoke_test.py --api http://SERVER_IP:8000 --out smoke.json
.\deploy\run_ui.ps1 -Api http://SERVER_IP:8000 -Port 8501
```

Check the model hash, schema, decoder and 10 trips in health, then select a trip in the client and show the Folium layers, both proxy scores, continuity, request timing and laptop 1 logs. Submit an unknown numeric trip ID directly to the API to demonstrate a 404. For the representative view, run `python -m ui.map_view --api http://SERVER_IP:8000 --trip 00000052 --out map.html`; keep the data-bearing HTML local unless distribution is approved. The API map deliberately excludes reference geometry. For a reference overlay in the report, use a separate local evaluation-only figure, not the reviewer API.

If client isolation blocks the LAN rehearsal, use an approved private hotspot or run both commands on localhost and label the result single-machine only. Ctrl+C in each hosting terminal shuts down that service; for background processes stop only the verified PathSnap process IDs.
