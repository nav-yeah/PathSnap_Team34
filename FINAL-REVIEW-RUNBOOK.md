# PathSnap final-review engineering checklist

| Workstream | Official source | Owner | Verification |
| --- | --- | --- | --- |
| Dataset understanding and leakage controls | Dataset Exploration §§3–8, pp. 1–4 | Chetha | Archive inventory, proxy alignment evidence, candidate tests |
| Five approaches, grouped evaluation and Bayesian tuning | Models and Training Guidelines §§2–7, pp. 1–4; Deliverables §§3–4, pp. 3–5 | Chhavi and model owners | Review 1 comparator; final histogram protocol, trials and outer/student metrics |
| Sequential reconstruction and comparison | Models and Training Guidelines §§8–9, pp. 4–5; Deliverables §5, pp. 5–6 | Navya review | Opt-in Viterbi plug-in, tests and frozen student route comparison |
| Folium map, API/UI and two laptops | Deliverables §§6–7, pp. 6–7; Models and Training Guidelines final evaluation, p. 6 | Aryan | HTTP smoke test plus separate physical LAN rehearsal |
| Single report, artifacts and integrity | Deliverables §§1, 8; Assessment and Plagiarism Policy §§3–7 | Team | Technical evidence here; team writes final report independently |

The documents describe workstreams rather than the four numbered phase definitions used in the request. Follow their actual requirements. Candidate metrics are not route accuracy. The decoder allows one directed bridge arc between observed states and starts a new chain when no transition is plausible. It outputs one arc per observation and does not stitch intermediate arcs into a full route. Direct adjacency is therefore stricter than bridge reachability but does not certify a reconstructed route.

## Reproduce and rehearse

```powershell
py -3.12 -m pip install -r requirements-dev.txt
py -3.12 -m pytest -q
py -3.12 -m pathsnap_ml.final_histogram "$HOME/Downloads/Student folder.zip" final-run
py -3.12 evaluate_routes.py "$HOME/Downloads/Student folder.zip" final-run/model.joblib final-run/routes.json
$env:PATHSNAP_MODEL = (Resolve-Path final-run/model.joblib).Path
$env:PATHSNAP_TRIPS_DIR = 'C:\path\to\student_test'
$env:PATHSNAP_REFERENCE_DIR = $env:PATHSNAP_TRIPS_DIR
$env:PATHSNAP_DECODER = 'pathsnap_ml.viterbi:decode'
py -3.12 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

On laptop 2, use `deploy/run_ui.ps1` with laptop 1's LAN URL (see `docs/DEPLOYMENT.md`). Start with `GET /health` to show model version, parameters and decoder; run `py -3.12 deploy/smoke_test.py --api http://SERVER_IP:8000 --out smoke.json`. In Streamlit select a trip and compare classifier and decoder proxy Top-1, directed adjacency, unmatched counts, and latency. Toggle the Folium layers to inspect GPS and candidate/predicted arcs. Keep evaluator references server-side. If campus Wi-Fi isolates clients, use an approved hotspot and a narrow private-network firewall rule; if necessary, rehearse on localhost and label it a single-machine rehearsal.

## Viva quick checks

- **Why not zip track and route rows?** They represent sampled observations and traversed arc occurrences, respectively. Multiple GPS samples can land on one arc; other arcs can be skipped by sampling.
- **What is Top-1?** Among accepted proxy-labeled observations, the fraction whose top-scored candidate equals the proxy arc. Candidate misses fail; uncertain labels are excluded.
- **Why nested grouping?** Inner trip groups select hyperparameters, outer groups assess the search procedure; student trips are used after freezing the artifact.
- **Why histogram boosting?** Team selection based on near-leading ranking, strongest candidate macro F1 and faster measured fitting than Random Forest; not a significance claim.
- **What does the decoder do?** Adds clipped log compatibility scores to penalties for disagreement between directed path length and GPS displacement, stores backpointers and traces each feasible chain. Scores are not calibrated likelihoods.
- **Where is inference?** The API on laptop 1 reads track, nodes and arcs, then scores and decodes; the client only displays results. Reference routes are loaded for scoring after prediction.
