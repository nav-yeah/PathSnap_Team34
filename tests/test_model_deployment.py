"""Final-model deployment: artifact description, version checks, promotion."""
import hashlib
import subprocess
import sys
import warnings

import joblib
import numpy as np
import pytest

from pathsnap_ml.data import FEATURES, SCHEMA, align, candidates
from pathsnap_ml.train import assemble, fit
from tests.conftest import ROOT, TRIP


@pytest.fixture(scope="session")
def hgb_model_path(dataset, tmp_path_factory):
    """Team histogram pipeline fitted on the synthetic trips (test-only artifact)."""
    items = []
    for trip, (track, nodes, arcs, route) in dataset[1].items():
        values, ids, points, starts, ends = candidates(track, nodes, arcs)
        items.append((trip, values, ids, align(points, starts, ends, route)))
    data = assemble(items)
    model, _ = fit("histogram", {"max_iter": 15}, data, np.arange(len(data[0])))
    path = tmp_path_factory.mktemp("hgb") / "histogram.joblib"
    joblib.dump(dict(schema=SCHEMA, features=FEATURES, version="test-histogram", model=model), path)
    return path


def test_health_describes_histogram_model(client_for, hgb_model_path):
    model = client_for(model_path=hgb_model_path).get("/health").json()["model"]
    assert model["family"] == "Histogram Gradient Boosting"
    assert model["estimator"] == "HistGradientBoostingClassifier"
    assert model["key_parameters"]["max_iter"] == 15
    assert 1 <= model["fitted_iterations"] <= 15
    assert model["sha256"] == hashlib.sha256(hgb_model_path.read_bytes()).hexdigest()
    assert model["features"] == list(FEATURES) and model["version_warnings"] == []


def test_histogram_model_serves_predictions(client_for, hgb_model_path):
    body = client_for(model_path=hgb_model_path).post("/predict", json={"trip_id": TRIP}).json()
    assert body["model_version"] == "test-histogram"
    assert len(body["predicted_arc_ids"]) == 60 and body["metrics"]["top1_accuracy"] >= 0.8


def test_sklearn_version_mismatch_is_reported(monkeypatch, hgb_model_path):
    from sklearn.exceptions import InconsistentVersionWarning

    import api.model_info as model_info
    real_load = model_info.load

    def load_with_warning(path):
        warnings.warn(InconsistentVersionWarning(
            estimator_name="HistGradientBoostingClassifier",
            current_sklearn_version="1.9.1", original_sklearn_version="1.5.0"))
        return real_load(path)

    monkeypatch.setattr(model_info, "load", load_with_warning)
    _, info = model_info.load_with_info(hgb_model_path)
    assert len(info["version_warnings"]) == 1 and "1.5.0" in info["version_warnings"][0]


def run_promote(*args):
    return subprocess.run([sys.executable, str(ROOT / "deploy" / "promote_model.py"), *map(str, args)],
                          capture_output=True, text=True, cwd=ROOT)


def test_promote_installs_and_verifies_histogram(hgb_model_path, tmp_path):
    destination = tmp_path / "artifacts" / "model.joblib"
    result = run_promote(hgb_model_path, "--dest", destination)
    assert result.returncode == 0, result.stdout + result.stderr
    assert destination.read_bytes() == hgb_model_path.read_bytes()
    assert "OK: installed Histogram Gradient Boosting" in result.stdout


def test_promote_refuses_wrong_family(model_path, tmp_path):
    destination = tmp_path / "model.joblib"
    result = run_promote(model_path, "--dest", destination)
    assert result.returncode == 1 and "REFUSED" in result.stdout
    assert "Logistic Regression" in result.stdout and not destination.exists()


def test_promote_refuses_incompatible_schema(tmp_path):
    bad = tmp_path / "bad.joblib"
    joblib.dump(dict(schema="other", features=("x",), version="v", model=None), bad)
    result = run_promote(bad, "--dest", tmp_path / "out.joblib")
    assert result.returncode != 0 and "Incompatible artifact schema" in result.stderr


def test_ui_sidebar_shows_model_family(monkeypatch, client_for, hgb_model_path):
    from streamlit.testing.v1 import AppTest

    from ui.api_client import PathSnapClient

    api = client_for(model_path=hgb_model_path)
    monkeypatch.setattr(PathSnapClient, "health", lambda self: api.get("/health").json())
    monkeypatch.setattr(PathSnapClient, "trips", lambda self: api.get("/trips").json())
    app = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert any("Histogram Gradient Boosting" in item.value for item in app.sidebar.markdown)
