"""Discovery and loading of trip inputs for inference.

Only .track, .nodes and .arcs are ever read here. Reference routes live in
``api.evaluation`` so ground truth cannot reach the prediction pipeline.
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

TRIP_ID = re.compile(r"^[0-9]{1,16}$")
INFERENCE_SUFFIXES = (".track", ".nodes", ".arcs")


class TripNotFound(LookupError):
    pass


class InvalidTripData(ValueError):
    pass


@dataclass(frozen=True)
class TripInputs:
    trip_id: str
    track: np.ndarray   # (N, 3): longitude, latitude, timestamp [s]
    nodes: np.ndarray   # (M, 2): longitude, latitude
    arcs: np.ndarray    # (K, 2): from_node_id, to_node_id


class TripStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def trip_ids(self):
        if not self.root.is_dir():
            return []
        found = []
        for folder in sorted(self.root.iterdir()):
            if folder.is_dir() and TRIP_ID.match(folder.name) and all(
                (folder / f"{folder.name}{suffix}").is_file() for suffix in INFERENCE_SUFFIXES
            ):
                found.append(folder.name)
        return found

    def _folder(self, trip_id):
        # Whitelist against discovered trips: blocks path traversal and typos.
        if not isinstance(trip_id, str) or not TRIP_ID.match(trip_id) \
                or trip_id not in self.trip_ids():
            raise TripNotFound(trip_id)
        return self.root / trip_id

    def summary(self, trip_id):
        track = self.load(trip_id).track
        return dict(trip_id=trip_id, observations=int(len(track)),
                    duration_s=float(track[-1, 2] - track[0, 2]))

    def load(self, trip_id) -> TripInputs:
        folder = self._folder(trip_id)
        return _load_cached(str(folder), trip_id, _mtime(folder, trip_id))


def _mtime(folder, trip_id):
    return tuple((folder / f"{trip_id}{s}").stat().st_mtime_ns for s in INFERENCE_SUFFIXES)


@lru_cache(maxsize=32)
def _load_cached(folder, trip_id, _stamp):
    folder = Path(folder)

    def read(suffix, dtype, columns):
        path = folder / f"{trip_id}{suffix}"
        try:
            values = np.loadtxt(path, dtype=dtype, ndmin=2)
        except ValueError as error:
            raise InvalidTripData(f"{path.name}: {error}") from error
        if values.ndim != 2 or values.shape[1] != columns or len(values) == 0:
            raise InvalidTripData(f"{path.name}: expected {columns} columns, got {values.shape}")
        return values

    track = read(".track", float, 3)
    nodes = read(".nodes", float, 2)
    arcs = read(".arcs", np.int64, 2)
    if len(track) < 2:
        raise InvalidTripData("track needs at least two GPS observations")
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise InvalidTripData("non-finite coordinates or timestamps")
    lon, lat = track[:, 0], track[:, 1]
    if (np.abs(lon) > 180).any() or (np.abs(lat) > 90).any():
        raise InvalidTripData("coordinates outside longitude/latitude range")
    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise InvalidTripData("arc references a node id outside .nodes")
    for array in (track, nodes, arcs):
        array.setflags(write=False)
    return TripInputs(trip_id, track, nodes, arcs)
