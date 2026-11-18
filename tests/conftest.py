"""Synthetic fixtures: a tiny road network so API tests run without the
university dataset (which must not be committed)."""
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pathsnap_ml.data import FEATURES, SCHEMA, align, candidates  # noqa: E402
from pathsnap_ml.train import assemble, fit  # noqa: E402

TRIP = "00000001"
SECOND_TRIP = "00000002"


def _network():
    """Two parallel east-west roads (~45 m apart) with arcs in both directions,
    joined by a connector (around 40.18 N, 44.50 E)."""
    lon = 44.50 + np.arange(30) * 0.0002            # ~17 m spacing
    north = np.column_stack((lon, np.full(30, 40.1800)))
    south = np.column_stack((lon, np.full(30, 40.1796)))
    nodes = np.vstack((north, south))
    forward = [(i, i + 1) for i in range(29)]
    arcs = forward + [(b, a) for a, b in forward]
    arcs += [(30 + a, 30 + b) for a, b in forward] + [(30 + b, 30 + a) for a, b in forward]
    arcs += [(15, 45), (45, 15)]
    return nodes, np.array(arcs)


def _write_trip(folder, trip_id, rng, with_route=True):
    nodes, arcs = _network()
    t = np.arange(60, dtype=float)
    lon = 44.5002 + t * (0.0002 * 26 / 60)
    lat = 40.1800 + rng.normal(0, 0.00002, len(t))   # ~2 m noise on the north road
    track = np.column_stack((lon + rng.normal(0, 0.00002, len(t)), lat, t))
    route = np.arange(1, 28)                         # eastbound north-road arcs
    directory = folder / trip_id
    directory.mkdir(parents=True)
    np.savetxt(directory / f"{trip_id}.track", track, fmt="%.7f")
    np.savetxt(directory / f"{trip_id}.nodes", nodes, fmt="%.7f")
    np.savetxt(directory / f"{trip_id}.arcs", arcs, fmt="%d")
    if with_route:
        np.savetxt(directory / f"{trip_id}.route", route, fmt="%d")
    return track, nodes, arcs, route


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    folder = tmp_path_factory.mktemp("trips")
    rng = np.random.default_rng(34)
    raw = {trip: _write_trip(folder, trip, rng) for trip in (TRIP, SECOND_TRIP)}
    return folder, raw


@pytest.fixture(scope="session")
def model_path(dataset, tmp_path_factory):
    """Fit the team's logistic pipeline on the synthetic trips (test-only artifact)."""
    _, raw = dataset
    items = []
    for trip, (track, nodes, arcs, route) in raw.items():
        values, ids, points, starts, ends = candidates(track, nodes, arcs)
        items.append((trip, values, ids, align(points, starts, ends, route)))
    data = assemble(items)
    model, _ = fit("logistic", None, data, np.arange(len(data[0])))
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(dict(schema=SCHEMA, features=FEATURES, version="test-logistic", model=model), path)
    return path


@pytest.fixture
def settings_for(dataset, model_path, tmp_path):
    from api.config import Settings

    def make(**overrides):
        values = dict(model_path=model_path, trips_dir=dataset[0], reference_dir=dataset[0],
                      decoder="classifier_argmax", cors_origins=("*",), max_context_arcs=6000)
        return Settings(**(values | overrides))
    return make


@pytest.fixture
def client_for(settings_for):
    from fastapi.testclient import TestClient

    from api.main import create_app

    clients = []

    def make(**overrides):
        client = TestClient(create_app(settings_for(**overrides)))
        client.__enter__()
        clients.append(client)
        return client
    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def copy_trips(dataset, tmp_path):
    def make(drop_suffix=None):
        target = tmp_path / "copy"
        shutil.copytree(dataset[0], target)
        if drop_suffix:
            for path in target.rglob(f"*{drop_suffix}"):
                path.unlink()
        return target
    return make
