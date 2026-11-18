"""Server-side scoring against reference routes.

This is the ONLY module that opens .route files. It runs after prediction is
complete, receives the finished prediction, and returns aggregate numbers only:
no reference arc ids, geometry or per-observation correctness leave the server.
"""
from pathlib import Path

import numpy as np

from pathsnap_ml.data import align

from .pipeline import Prediction
from .trip_store import TRIP_ID


class ReferenceStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def route(self, trip_id):
        if not TRIP_ID.match(trip_id):
            return None
        path = self.root / trip_id / f"{trip_id}.route"
        if not path.is_file():
            return None
        route = np.loadtxt(path, dtype=np.int64, ndmin=1)
        return route if route.size else None


def score(prediction: Prediction, route) -> dict:
    context = prediction.context
    if route.min() < 0 or route.max() >= len(context.arcs):
        raise ValueError("reference route contains arc ids outside the road network")
    # Same weak reference-proxy labelling the team uses offline (35 m / 8 m margin).
    reference = align(context.points_m, context.starts_m, context.ends_m, route)
    accepted = reference >= 0
    count = int(accepted.sum())

    def top1(arcs):
        return float((arcs[accepted] == reference[accepted]).mean()) if count else None

    hits = (prediction.candidate_arcs == reference[:, None]).any(axis=1)
    predicted = set(prediction.stitched_route)
    truth = set(int(arc) for arc in route)
    overlap = len(predicted & truth)
    return dict(
        top1_accuracy=top1(prediction.decoded_arcs),
        classifier_top1_accuracy=top1(prediction.classifier_arcs),
        candidate_recall=float(hits[accepted].mean()) if count else None,
        label_coverage=float(accepted.mean()),
        scored_observations=count,
        route_arc_precision=overlap / len(predicted) if predicted else None,
        route_arc_recall=overlap / len(truth) if truth else None,
    )
