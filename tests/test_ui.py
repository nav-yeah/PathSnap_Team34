"""UI-side tests: HTTP client error handling, Folium map, headless Streamlit run."""
from pathlib import Path

import pytest
import requests

from tests.conftest import ROOT, TRIP
from ui.api_client import ApiError, PathSnapClient
from ui.map_view import build_map


@pytest.fixture
def prediction(client_for):
    return client_for().post("/predict", json={"trip_id": TRIP}).json()


def test_client_reports_unreachable_api_clearly():
    with pytest.raises(ApiError, match="Cannot reach the API"):
        PathSnapClient("http://127.0.0.1:9", timeout=(1, 1)).health()


def test_client_surfaces_http_error_detail(monkeypatch):
    class Response:
        ok, status_code, text = False, 404, "nope"

        def json(self):
            return {"detail": "unknown trip_id '1'"}

    monkeypatch.setattr(requests, "request", lambda *a, **k: Response())
    with pytest.raises(ApiError, match=r"HTTP 404: unknown trip_id"):
        PathSnapClient("http://x").predict("1")


def test_map_has_all_layers_and_legend(prediction):
    html = build_map(prediction).get_root().render()
    for name in ("Road network (candidate arcs)", "Observed GPS track", "Predicted route",
                 f"<b>Trip {TRIP}</b>"):
        assert name in html
    assert "reference" not in html.lower()


def test_streamlit_app_runs_end_to_end(monkeypatch, client_for):
    from streamlit.testing.v1 import AppTest

    api = client_for()
    monkeypatch.setattr(PathSnapClient, "health", lambda self: api.get("/health").json())
    monkeypatch.setattr(PathSnapClient, "trips", lambda self: api.get("/trips").json())
    monkeypatch.setattr(PathSnapClient, "predict",
                        lambda self, trip: api.post("/predict", json={"trip_id": trip}).json())
    app = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert app.selectbox[0].options == [f"Trip {TRIP} · 60 GPS points", "Trip 00000002 · 60 GPS points"]
    next(b for b in app.button if b.label == "Reconstruct route").click().run()
    assert not app.exception
    labels = {metric.label: metric.value for metric in app.metric}
    assert labels["Top-1 accuracy"].endswith("%") and labels["GPS points"] == "60"


def test_streamlit_shows_error_when_api_down(monkeypatch):
    from streamlit.testing.v1 import AppTest

    def down(self):
        raise ApiError("Cannot reach the API at http://10.0.0.9:8000.")
    monkeypatch.setattr(PathSnapClient, "health", down)
    app = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert any("Cannot reach the API" in error.value for error in app.error)
