# PathSnap — Team 34

Distributed map matching and route reconstruction. This proposed baseline contains the shared ML schema and evaluation tools, plus the FastAPI backend, Streamlit evaluator UI and two-laptop deployment. Teammate preprocessing and the HMM decoder are not included yet.

See [ML setup and interfaces](pathsnap_ml/README.md) and [API, UI and deployment](docs/DEPLOYMENT.md). The university dataset is not redistributed. Review 1 results are generated locally, not assumed by the source code.

| Folder | Contents | Owner |
|---|---|---|
| `pathsnap_ml/` | Loading, alignment, candidates, features, model training/comparison, Optuna | Chhavi / Chetha |
| `api/` | FastAPI server: trip loading, inference pipeline, decoder plug-in, server-side scoring | Aryan |
| `ui/` | Streamlit evaluator UI and Folium route map | Aryan |
| `deploy/` | Laptop 1 / Laptop 2 launch scripts, firewall helper, smoke test, final-model promotion | Aryan |
| `tests/` | API and UI integration tests (synthetic data) | Aryan |
