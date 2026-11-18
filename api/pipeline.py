"""Reference-free inference pipeline: trip inputs -> candidates -> scores -> route.

Uses the team's shared modules unchanged so offline and deployed scoring match:
``pathsnap_ml.data.candidates`` (candidate generation and features) and
``pathsnap_ml.inference.predict`` (artifact contract).
"""
import time
import heapq
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
    stitched_route: list[int]
    route_break_indices: list[int]
    inserted_bridge_arcs: int
    gps_gap_breaks: int
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
    clock = time.perf_counter()
    stitched, breaks, bridges, gap_breaks = stitch_route(
        decoded, trip.track[:, 2], trip.arcs, points, starts, ends)
    timings["stitch_s"] = time.perf_counter() - clock
    return Prediction(trip.trip_id, arc_ids, scores, classifier, decoded, stitched, breaks,
                      bridges, gap_breaks, context, timings)


def _shortest_bridge(source, target, outgoing, destinations, lengths, max_distance):
    if source == target:
        return []
    distances = {int(source): 0.0}
    previous = {}
    queue = [(0.0, int(source))]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == target:
            path = []
            while node != source:
                node, arc = previous[node]
                path.append(arc)
            return path[::-1]
        for arc in outgoing[node]:
            next_distance = distance + lengths[arc]
            next_node = int(destinations[arc])
            if next_distance <= max_distance and next_distance < distances.get(next_node, np.inf):
                distances[next_node] = next_distance
                previous[next_node] = (node, arc)
                heapq.heappush(queue, (next_distance, next_node))
    return None


def stitch_route(decoded_arcs, timestamps, arcs, points_m, starts_m, ends_m,
                 *, max_bridge_m=2000.0, gap_s=30.0):
    """Expand decoded observations into directed arcs; leave explicit breaks at gaps/unreachable joins."""
    decoded_arcs = np.asarray(decoded_arcs, dtype=np.int64)
    timestamps = np.asarray(timestamps, dtype=float)
    arcs = np.asarray(arcs, dtype=np.int64)
    if decoded_arcs.ndim != 1 or timestamps.shape != decoded_arcs.shape:
        raise ValueError("decoded_arcs and timestamps must be matching one-dimensional arrays")
    if arcs.ndim != 2 or arcs.shape[1] != 2:
        raise ValueError("arcs must have shape (K, 2)")
    if decoded_arcs.size == 0 or arcs.shape[0] == 0:
        return [], [], 0, 0

    lengths = np.linalg.norm(np.asarray(ends_m) - np.asarray(starts_m), axis=1)
    outgoing = [[] for _ in range(int(arcs.max()) + 1)]
    for arc_id, (source, _) in enumerate(arcs):
        outgoing[int(source)].append(arc_id)
    for choices in outgoing:
        choices.sort()

    route, breaks = [], []
    inserted, gap_breaks = 0, 0
    previous_observation = None
    for observation, arc_id in enumerate(decoded_arcs):
        if arc_id < 0:
            previous_observation = None
            continue
        if previous_observation is None:
            if route and (not breaks or breaks[-1] != len(route)):
                breaks.append(len(route))
            route.append(int(arc_id))
            previous_observation = observation
            continue
        previous_arc = int(decoded_arcs[previous_observation])
        elapsed = timestamps[observation] - timestamps[previous_observation]
        if elapsed <= 0 or elapsed > gap_s:
            gap_breaks += 1
            if not breaks or breaks[-1] != len(route):
                breaks.append(len(route))
            route.append(int(arc_id))
        elif previous_arc == arc_id:
            pass
        elif arcs[previous_arc, 1] == arcs[arc_id, 0]:
            route.append(int(arc_id))
        else:
            bridge = _shortest_bridge(int(arcs[previous_arc, 1]), int(arcs[arc_id, 0]),
                                      outgoing, arcs[:, 1], lengths, max_bridge_m)
            if bridge is None:
                if not breaks or breaks[-1] != len(route):
                    breaks.append(len(route))
            else:
                route.extend(bridge)
                inserted += len(bridge)
            route.append(int(arc_id))
        previous_observation = observation
    return route, breaks, inserted, gap_breaks


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


def stitched_continuity(route, break_indices, arcs):
    if len(route) < 2:
        return dict(continuity=1.0, transitions=0, disconnected_jumps=0)
    breaks = set(break_indices)
    pairs = [(left, right) for index, (left, right) in enumerate(zip(route, route[1:]), start=1)
             if index not in breaks]
    if not pairs:
        return dict(continuity=1.0, transitions=0, disconnected_jumps=0)
    connected = sum(arcs[left, 1] == arcs[right, 0] for left, right in pairs)
    return dict(continuity=connected / len(pairs), transitions=len(pairs),
                disconnected_jumps=len(pairs) - int(connected))


def segment_coords(arc_ids, nodes, arcs):
    """[[lat, lon], [lat, lon]] per arc, for map display."""
    ends = nodes[arcs[np.asarray(arc_ids, dtype=np.int64)]]   # (n, 2 nodes, lon/lat)
    return np.round(ends[..., ::-1], 7).tolist()
