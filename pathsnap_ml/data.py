from pathlib import PurePosixPath

import numpy as np
from scipy.spatial import cKDTree

FEATURES = ("distance_m", "heading_cos", "fraction", "length_m", "speed_mps")
SCHEMA = "pathsnap-ml-v1"


def geometry(points, starts, ends):
    delta = ends - starts
    squared = np.sum(delta ** 2, axis=-1)
    fraction = np.clip(np.sum((points - starts) * delta, axis=-1)
                       / np.maximum(squared, 1e-12), 0, 1)
    distance = np.linalg.norm(points - starts - fraction[..., None] * delta, axis=-1)
    return distance, fraction, delta, np.sqrt(squared)


def align(points, starts, ends, route, radius=35, margin=8):
    route = np.asarray(route, dtype=int)
    if len(route) == 0:
        raise ValueError("Empty reference route")
    start, end = starts[route], ends[route]
    distance = np.stack([geometry(point, start, end)[0] for point in points]).astype("float32")
    cost = np.minimum(distance, 100)
    forward = np.empty_like(cost)
    forward[0] = cost[0]
    for index in range(1, len(points)):
        forward[index] = cost[index] + np.minimum.accumulate(forward[index - 1])
    result = np.full(len(points), -1, dtype=int)
    tail = np.zeros(len(route), dtype=np.float32)
    for index in range(len(points) - 1, -1, -1):
        total = forward[index] + tail
        best = int(np.argmin(total))
        alternatives = total[route != route[best]]
        if distance[index, best] <= radius and (
            not alternatives.size or alternatives.min() - total[best] >= margin
        ):
            result[index] = route[best]
        tail = np.minimum.accumulate((cost[index] + tail)[::-1])[::-1]
    return result


def candidates(track, nodes, arcs, count=8):
    origin = np.median(track[:, :2], axis=0)
    scale = 111195 * np.array([np.cos(np.deg2rad(origin[1])), 1])
    points = (track[:, :2] - origin) * scale
    starts, ends = ((nodes - origin) * scale)[arcs].transpose(1, 0, 2)
    length = np.linalg.norm(ends - starts, axis=1)
    subdivisions = np.maximum(1, np.ceil(length / 20).astype(int))
    owners = np.repeat(np.arange(len(arcs)), subdivisions + 1)
    offsets = np.cumsum(subdivisions + 1) - subdivisions - 1
    fraction = (np.arange(len(owners)) - offsets[owners]) / subdivisions[owners]
    samples = starts[owners] + fraction[:, None] * (ends - starts)[owners]
    tree = cKDTree(samples)
    _, neighbors = tree.query(points, k=min(64, len(samples)))
    _, fallback = cKDTree((starts + ends) / 2).query(points, k=min(count, len(arcs)))
    movement = np.diff(points, axis=0, prepend=points[:1])
    elapsed = np.diff(track[:, 2], prepend=track[0, 2])
    rows, ids = [], []
    for index, point in enumerate(points):
        pool = np.union1d(owners[neighbors[index]], fallback[index])
        distance, frac, delta, lengths = geometry(point, starts[pool], ends[pool])
        pick = np.argsort(distance, kind="stable")[:count]
        step = np.linalg.norm(movement[index])
        heading = (delta @ movement[index]) / np.maximum(lengths * step, 1e-12)
        if step < 2 or elapsed[index] <= 0:
            heading[:] = np.nan
        speed = step / elapsed[index] if elapsed[index] > 0 else np.nan
        rows.append(np.column_stack((distance, heading, frac, lengths,
                                     np.full(len(pool), speed)))[pick])
        ids.append(pool[pick])
    return np.stack(rows), np.stack(ids), points, starts, ends


def trips(archive, partition):
    if partition not in {"train", "student_test"}:
        raise ValueError("Only train and student_test are permitted")
    for name in sorted(archive.namelist()):
        path = PurePosixPath(name)
        if len(path.parts) != 4 or path.parts[1] != partition or path.suffix != ".track":
            continue
        def read(suffix, dtype=float, ndmin=2):
            with archive.open(str(path.with_suffix(suffix))) as stream:
                return np.loadtxt(stream, dtype=dtype, ndmin=ndmin)
        track, nodes, arcs = read(".track"), read(".nodes"), read(".arcs", int)
        route = read(".route", int, 1)
        features, ids, points, starts, ends = candidates(track, nodes, arcs)
        yield path.stem, features, ids, align(points, starts, ends, route)
