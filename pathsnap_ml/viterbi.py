"""Reviewable decoder plug-in; does not replace the API default."""
from collections import defaultdict

import numpy as np


def decode(scores, candidate_arcs, context):
    scores, ids = np.asarray(scores), np.asarray(candidate_arcs)
    if scores.ndim != 2 or ids.shape != scores.shape or len(ids) != len(context.track):
        raise ValueError("Invalid decoder input shapes")
    output = np.full(len(ids), -1, dtype=int)
    if not ids.size:
        return output
    arcs = context.arcs
    if not len(arcs):
        return output
    lengths = np.linalg.norm(context.ends_m - context.starts_m, axis=1)
    bridges = defaultdict(lambda: np.inf)
    for (start, end), length in zip(arcs, lengths, strict=True):
        key = (int(start), int(end))
        bridges[key] = min(bridges[key], length)
    valid = (ids >= 0) & (ids < len(arcs)) & np.isfinite(scores)
    emission = np.where(valid, np.log(np.clip(scores, 1e-9, 1)), -np.inf)
    safe = np.clip(ids, 0, len(arcs) - 1)
    starts, ends = context.starts_m[safe], context.ends_m[safe]
    delta = ends - starts
    fraction = np.clip(np.sum((context.points_m[:, None] - starts) * delta, axis=2)
                       / np.maximum(np.sum(delta ** 2, axis=2), 1e-12), 0, 1)
    progress = fraction * lengths[safe]
    parents = np.full(ids.shape, -1, dtype=int)
    previous = np.full(ids.shape[1], -np.inf)

    def trace(index, potentials):
        if index < 0 or not np.isfinite(potentials).any():
            return
        state = int(potentials.argmax())
        while index >= 0 and state >= 0:
            output[index] = ids[index, state]
            state = parents[index, state]
            index -= 1

    for index in range(len(ids)):
        current = emission[index].copy()
        if index and np.isfinite(previous).any() and valid[index].any():
            left, right = safe[index - 1], safe[index]
            elapsed = context.track[index, 2] - context.track[index - 1, 2]
            displacement = np.linalg.norm(context.points_m[index] - context.points_m[index - 1])
            end_nodes, start_nodes = arcs[left, 1], arcs[right, 0]
            bridge = np.array([[0 if end == start else bridges.get((end, start), np.inf)
                                for start in start_nodes] for end in end_nodes])
            distance = (lengths[left] - progress[index - 1])[:, None] + bridge + progress[index]
            same = left[:, None] == right
            forward = progress[index] - progress[index - 1, :, None]
            distance = np.where(same & (forward >= -12), np.maximum(forward, 0), distance)
            allowed = (elapsed > 0) & (elapsed <= 30) & (distance <= 55 * elapsed + 25)
            options = previous[:, None] - np.where(
                allowed, np.abs(distance - displacement) / max(20, displacement), np.inf)
            winners = options.argmax(axis=0)
            best = options[winners, np.arange(len(right))]
            continued = current + best
            if np.isfinite(continued).any():
                parents[index] = np.where(np.isfinite(continued), winners, -1)
                previous = continued
                continue
        trace(index - 1, previous)
        previous = current
    trace(len(ids) - 1, previous)
    return output
