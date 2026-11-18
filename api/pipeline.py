"""Reference-free inference pipeline: trip inputs -> candidates -> scores -> route.

Uses the team's shared modules unchanged so offline and deployed scoring match:
``pathsnap_ml.data.candidates`` (candidate generation and features) and
``pathsnap_ml.inference.predict`` (artifact contract).
"""
import time
from dataclasses import dataclass

import numpy as np

from pathsnap_ml.data import FEATURES, candidates
from pathsnap_ml.inference import predict

from .decoders import DecodeContext, classifier_argmax
from .trip_store import TripInputs


@dataclass
class Prediction:
    trip_id: str
    candidate_arcs: np.ndarray      # (N, C)
    scores: np.ndarray              # (N, C)
    classifier_arcs: np.ndarray     # (N,) classifier-only argmax
    decoded_arcs: np.ndarray        # (N,) after the configured decoder
    context: DecodeContext
    timings: dict


def run(trip: TripInputs, artifact, decoder) -> Prediction:
    timings = {}
    clock = time.perf_counter()
    features, arc_ids, points, starts, ends = candidates(trip.track, trip.nodes, trip.arcs)
    timings["candidates_s"] = time.perf_counter() - clock

    clock = time.perf_counter()
    rows = [dict(trip_id=trip.trip_id, observation_index=index,
                 timestamp=float(trip.track[index, 2]), arc_id=int(arc),
                 features=dict(zip(FEATURES, (None if not np.isfinite(v) else float(v)
                                              for v in vector), strict=True)))
            for index in range(len(arc_ids))
            for arc, vector in zip(arc_ids[index], features[index], strict=True)]
    scored = predict(artifact, rows)
    scores = np.array([row["score"] for row in scored]).reshape(arc_ids.shape)
    timings["scoring_s"] = time.perf_counter() - clock

    context = DecodeContext(trip.track, trip.nodes, trip.arcs, points, starts, ends)
    classifier = classifier_argmax(scores, arc_ids, context)
    clock = time.perf_counter()
    decoded = np.asarray(decoder(scores, arc_ids, context), dtype=np.int64)
    timings["decode_s"] = time.perf_counter() - clock
    if decoded.shape != (len(arc_ids),):
        raise ValueError(f"decoder returned shape {decoded.shape}, expected ({len(arc_ids)},)")
    if ((decoded < -1) | (decoded >= len(trip.arcs))).any():
        raise ValueError("decoder returned an arc id outside the road network")
    return Prediction(trip.trip_id, arc_ids, scores, classifier, decoded, context, timings)


def collapse(arc_sequence):
    """Ordered traversed arcs: drop unmatched (-1) and consecutive repeats."""
    route = []
    for arc in arc_sequence:
        if arc >= 0 and (not route or route[-1] != arc):
            route.append(int(arc))
    return route


def continuity(arc_sequence, arcs):
    """Share of arc changes where the next arc starts at the previous arc's end node.

    Reference-free: measures whether the predicted route is drivable on the
    directed network without teleporting between unconnected segments.
    """
    route = collapse(arc_sequence)
    transitions = len(route) - 1
    if transitions <= 0:
        return dict(continuity=1.0, transitions=0, disconnected_jumps=0)
    connected = sum(arcs[a, 1] == arcs[b, 0] for a, b in zip(route, route[1:]))
    return dict(continuity=connected / transitions, transitions=transitions,
                disconnected_jumps=transitions - int(connected))


def segment_coords(arc_ids, nodes, arcs):
    """[[lat, lon], [lat, lon]] per arc, for map display."""
    ends = nodes[arcs[np.asarray(arc_ids, dtype=np.int64)]]   # (n, 2 nodes, lon/lat)
    return np.round(ends[..., ::-1], 7).tolist()
