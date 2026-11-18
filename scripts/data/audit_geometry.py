"""Training-only input audit with deterministic exhaustive candidate checks."""
import argparse
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathsnap_ml.data import align, candidates, geometry
from pathsnap_ml.review1 import save


def audit(source, output):
    output.mkdir(parents=True)
    records, samples = {}, []
    with ZipFile(source) as archive:
        tracks = [PurePosixPath(name) for name in sorted(archive.namelist())
                  if name.endswith(".track")]
        for path in tracks:
            if len(path.parts) != 4 or path.parts[1] != "train":
                continue

            def read(suffix, dtype, ndmin=2):
                with archive.open(str(path.with_suffix(suffix))) as stream:
                    return np.loadtxt(stream, dtype=dtype, ndmin=ndmin)

            track, nodes = read(".track", float), read(".nodes", float)
            arcs, route = read(".arcs", int), read(".route", int, 1)
            if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
                raise ValueError(f"Invalid shape: {path}")
            if not np.isfinite(track).all() or not np.isfinite(nodes).all():
                raise ValueError(f"Nonfinite coordinates: {path}")
            for coordinates in (track[:, :2], nodes):
                if (np.abs(coordinates) > [180, 90]).any():
                    raise ValueError(f"Coordinate out of range: {path}")
            if ((arcs < 0) | (arcs >= len(nodes))).any() or ((route < 0) | (route >= len(arcs))).any():
                raise ValueError(f"Broken road reference: {path}")
            features, ids, points, starts, ends = candidates(track, nodes, arcs)
            proxy = align(points, starts, ends, route)
            nearest_misses = 0
            indices = np.linspace(0, len(track)-1, num=min(6, len(track)), dtype=int)
            for index in indices:
                distances = geometry(points[index], starts, ends)[0]
                nearest_misses += int(distances[ids[index]].min() > distances.min() + 1e-8)
            intervals = np.diff(track[:, 2])
            records[path.stem] = dict(
                observations=len(track), accepted=int((proxy >= 0).sum()),
                candidate_misses=int(((proxy >= 0) & ~(ids == proxy[:, None]).any(axis=1)).sum()),
                exhaustive_sample=len(indices), nearest_misses=nearest_misses,
                nonpositive_interval=int((intervals <= 0).sum()),
                duplicate_track_rows=len(track)-len(np.unique(track, axis=0)),
                degenerate_arcs=int((starts == ends).all(axis=1).sum()),
                speed_over_55=int((features[:, 0, 4] > 55).sum()))
            samples.append(features[:, 0, :])
            if len(samples) == 1:
                figure, axis = plt.subplots()
                from matplotlib.collections import LineCollection
                axis.add_collection(LineCollection(np.stack((starts, ends), axis=1),
                                                  colors="grey", linewidths=.3))
                axis.plot(*points.T, color="blue", linewidth=1)
                axis.set(aspect="equal", xlabel="Local east (m)", ylabel="Local north (m)")
                figure.savefig(output / "track-overlay.png", dpi=120)
    save(output / "dataset.json", dict(training_trips=len(records),
         student_track_inventory=sum(path.parts[1] == "student_test" for path in tracks
                                     if len(path.parts) == 4), trips=records))
    values = np.concatenate(samples)
    figure, axes = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
    for axis, column, title in zip(axes.flat, [0, 4, 1],
                                  ["Nearest distance (m)", "GPS speed (m/s)", "Heading cosine"]):
        data = values[:, column]
        axis.hist(data[np.isfinite(data)], bins=60, log=True)
        axis.set_title(title)
    total = sum(row["observations"] for row in records.values())
    accepted = sum(row["accepted"] for row in records.values())
    misses = sum(row["candidate_misses"] for row in records.values())
    axes[1, 0].bar(["Accepted", "Excluded"], [accepted, total-accepted])
    axes[1, 1].hist([row["accepted"]/row["observations"] for row in records.values()])
    axes[1, 1].set_title("Alignment coverage by trip")
    axes[1, 2].hist([row["observations"] for row in records.values()])
    axes[1, 2].set_title(f"GPS count by trip; candidate recall {1-misses/accepted:.2%}")
    figure.savefig(output / "distributions.png", dpi=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.archive, args.output)
