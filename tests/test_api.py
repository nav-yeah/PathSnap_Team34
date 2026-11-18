"""API integration tests (FastAPI TestClient, synthetic road network)."""
import json

import numpy as np
import pytest

from tests.conftest import SECOND_TRIP, TRIP

REFERENCE_WORDS = ("reference", "ground_truth", "truth", "route_file", "aligned")


def test_health_reports_model_decoder_and_trips(client_for):
    body = client_for().get("/health").json()
    assert body == dict(status="ok", model_loaded=True, model_version="test-logistic",
                        model_error=None, decoder="classifier_argmax", trips_available=2)


def test_trip_listing_contains_only_reference_free_summaries(client_for):
    trips = client_for().get("/trips").json()
    assert [t["trip_id"] for t in trips] == [TRIP, SECOND_TRIP]
    assert all(set(t) == {"trip_id", "observations", "duration_s"} for t in trips)
    assert trips[0]["observations"] == 60 and trips[0]["duration_s"] == 59.0


def test_predict_returns_route_metrics_and_geometry(client_for):
    response = client_for().post("/predict", json={"trip_id": TRIP})
    assert response.status_code == 200
    body = response.json()
    assert body["observations"] == len(body["predicted_arc_ids"]) == 60
    assert len(body["geometry"]["gps_points"]) == 60
    assert len(body["geometry"]["predicted_route"]) == len(body["route_arc_ids"]) > 0
    lat, lon = body["geometry"]["gps_points"][0]
    assert 40 < lat < 41 and 44 < lon < 45          # [lat, lon] order for Folium
    metrics = body["metrics"]
    assert body["evaluation_available"] is True
    assert metrics["top1_accuracy"] >= 0.8           # easy synthetic trip
    assert metrics["candidate_recall"] == 1.0
    assert 0 <= metrics["route_arc_precision"] <= 1 and 0 <= metrics["route_arc_recall"] <= 1
    assert body["route_quality"]["continuity"] >= 0.8
    assert {"candidates_s", "scoring_s", "decode_s", "evaluation_s", "total_s"} <= set(body["timings_ms"])


def test_response_never_contains_reference_route(client_for, dataset):
    body = client_for().post("/predict", json={"trip_id": TRIP}).json()
    text = json.dumps(body).lower()
    assert not any(word in text for word in REFERENCE_WORDS)
    assert set(body) == {"trip_id", "model_version", "decoder", "observations",
                         "predicted_arc_ids", "route_arc_ids", "route_quality",
                         "evaluation_available", "metrics", "timings_ms", "geometry"}
    assert set(body["geometry"]) == {"gps_points", "predicted_route", "road_context"}
    assert set(body["metrics"]) == {"top1_accuracy", "classifier_top1_accuracy", "candidate_recall",
                                    "label_coverage", "scored_observations",
                                    "route_arc_precision", "route_arc_recall"}


def test_prediction_is_identical_with_and_without_reference_data(client_for, tmp_path):
    with_ref = client_for().post("/predict", json={"trip_id": TRIP}).json()
    empty = tmp_path / "no-references"
    empty.mkdir()
    without_ref = client_for(reference_dir=empty).post("/predict", json={"trip_id": TRIP}).json()
    assert with_ref["predicted_arc_ids"] == without_ref["predicted_arc_ids"]
    assert with_ref["route_arc_ids"] == without_ref["route_arc_ids"]
    assert without_ref["evaluation_available"] is False and without_ref["metrics"] is None


def test_inference_path_never_reads_route_files(settings_for, monkeypatch):
    import api.trip_store as trip_store
    from api import pipeline
    from api.decoders import classifier_argmax
    from pathsnap_ml.inference import load

    opened = []
    original = np.loadtxt
    monkeypatch.setattr(trip_store.np, "loadtxt",
                        lambda path, *a, **k: opened.append(str(path)) or original(path, *a, **k))
    trip_store._load_cached.cache_clear()
    settings = settings_for()
    trip = trip_store.TripStore(settings.trips_dir).load(TRIP)
    pipeline.run(trip, load(settings.model_path), classifier_argmax)
    assert opened and not any(path.endswith(".route") for path in opened)
    trip_store._load_cached.cache_clear()


def test_trips_without_route_files_still_predict(client_for, copy_trips):
    folder = copy_trips(drop_suffix=".route")
    client = client_for(trips_dir=folder, reference_dir=folder)
    body = client.post("/predict", json={"trip_id": TRIP}).json()
    assert body["evaluation_available"] is False and len(body["predicted_arc_ids"]) == 60


@pytest.mark.parametrize("payload,status", [
    ({"trip_id": "99999999"}, 404),
    ({"trip_id": "../00000001"}, 422),
    ({"trip_id": "00000001/../../etc"}, 422),
    ({"trip_id": ""}, 422),
    ({}, 422),
    ({"trip_id": TRIP, "route": [1, 2, 3]}, 422),     # extra fields rejected
])
def test_bad_requests_are_rejected(client_for, payload, status):
    assert client_for().post("/predict", json=payload).status_code == status


def test_missing_model_degrades_health_and_returns_503(client_for, tmp_path):
    client = client_for(model_path=tmp_path / "absent.joblib")
    health = client.get("/health").json()
    assert health["status"] == "degraded" and not health["model_loaded"]
    assert "not found" in health["model_error"]
    response = client.post("/predict", json={"trip_id": TRIP})
    assert response.status_code == 503 and "not found" in response.json()["detail"]


def test_incompatible_artifact_is_refused(client_for, tmp_path):
    import joblib
    path = tmp_path / "wrong.joblib"
    joblib.dump(dict(schema="other-schema", features=("x",), version="x", model=None), path)
    health = client_for(model_path=path).get("/health").json()
    assert not health["model_loaded"] and "Incompatible" in health["model_error"]


def test_corrupt_trip_file_returns_422(client_for, copy_trips):
    folder = copy_trips()
    (folder / TRIP / f"{TRIP}.track").write_text("44.5 40.1\nnot numbers here\n")
    response = client_for(trips_dir=folder).post("/predict", json={"trip_id": TRIP})
    assert response.status_code == 422 and "invalid data" in response.json()["detail"]


def test_arc_referencing_missing_node_returns_422(client_for, copy_trips):
    folder = copy_trips()
    with open(folder / SECOND_TRIP / f"{SECOND_TRIP}.arcs", "a") as stream:
        stream.write("0 99999\n")
    response = client_for(trips_dir=folder).post("/predict", json={"trip_id": SECOND_TRIP})
    assert response.status_code == 422


def test_plugin_decoder_is_used_and_reported(client_for):
    client = client_for(decoder="tests.plugin_decoders:first_candidate")
    body = client.post("/predict", json={"trip_id": TRIP}).json()
    assert body["decoder"] == "tests.plugin_decoders:first_candidate"
    assert client.get("/health").json()["decoder"] == body["decoder"]
    # Decoded and classifier-only accuracies are reported separately.
    assert body["metrics"]["classifier_top1_accuracy"] is not None


def test_misbehaving_decoder_returns_500_not_a_crash(client_for):
    client = client_for(decoder="tests.plugin_decoders:wrong_shape")
    response = client.post("/predict", json={"trip_id": TRIP})
    assert response.status_code == 500 and "decoder returned shape" in response.json()["detail"]
    assert client.get("/health").status_code == 200          # server still alive


def test_unmatched_observations_are_counted(client_for):
    body = client_for(decoder="tests.plugin_decoders:leave_half_unmatched").post(
        "/predict", json={"trip_id": TRIP}).json()
    assert body["route_quality"]["unmatched_observations"] == 30
    assert body["predicted_arc_ids"].count(-1) == 30


def test_bad_decoder_spec_fails_fast(settings_for):
    from api.decoders import resolve
    with pytest.raises(ValueError):
        resolve("not-a-module-spec")


def test_continuity_and_collapse():
    from api.pipeline import collapse, continuity
    arcs = np.array([[0, 1], [1, 2], [5, 6], [2, 3]])
    assert collapse([0, 0, -1, 1, 1, 3, 3]) == [0, 1, 3]
    assert continuity([0, 0, 1, 3], arcs) == dict(continuity=1.0, transitions=2, disconnected_jumps=0)
    assert continuity([0, 2, 3], arcs)["disconnected_jumps"] == 2
    assert continuity([1, 1], arcs)["continuity"] == 1.0
