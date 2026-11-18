"""Frozen-model student-partition evaluation; references loaded after prediction."""
import argparse
import time
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from threadpoolctl import threadpool_limits

from api.pipeline import continuity, run
from api.trip_store import TripInputs
from pathsnap_ml.data import align
from pathsnap_ml.inference import load
from pathsnap_ml.review1 import save
from pathsnap_ml.viterbi import decode


def evaluate(archive_path, model_path):
    artifact = load(model_path)
    results = {}
    with ZipFile(archive_path) as archive, threadpool_limits(limits=2):
        for name in sorted(archive.namelist()):
            path = PurePosixPath(name)
            if len(path.parts) != 4 or path.parts[1] != "student_test" or path.suffix != ".track":
                continue

            def read(suffix, dtype, dimensions=2):
                with archive.open(str(path.with_suffix(suffix))) as stream:
                    return np.loadtxt(stream, dtype=dtype, ndmin=dimensions)

            start = time.perf_counter()
            trip = TripInputs(path.stem, read(".track", float), read(".nodes", float),
                              read(".arcs", int))
            parsing = time.perf_counter() - start
            prediction = run(trip, artifact, decode)
            elapsed = time.perf_counter() - start
            context = prediction.context
            proxy = align(context.points_m, context.starts_m, context.ends_m,
                          read(".route", int, 1))
            accepted = proxy >= 0
            row = dict(observations=len(proxy), accepted=int(accepted.sum()),
                       timings=prediction.timings | dict(parse_s=parsing, total_s=elapsed))
            for label, arcs in [("classifier", prediction.classifier_arcs),
                                ("decoded", prediction.decoded_arcs)]:
                row[label] = continuity(arcs, trip.arcs) | dict(
                    correct=int(((arcs == proxy) & accepted).sum()),
                    unmatched=int((arcs < 0).sum()))
            results[path.stem] = row
    accepted = sum(row["accepted"] for row in results.values())
    totals = {}
    for label in ("classifier", "decoded"):
        correct = sum(row[label]["correct"] for row in results.values())
        transitions = sum(row[label]["transitions"] for row in results.values())
        jumps = sum(row[label]["disconnected_jumps"] for row in results.values())
        totals[label] = dict(top1=correct/accepted, correct=correct, transitions=transitions,
                             direct_adjacency=1-jumps/transitions if transitions else None,
                             unmatched=sum(row[label]["unmatched"] for row in results.values()))
    return dict(partition="student_test", accepted=accepted, model=artifact["version"],
                totals=totals, trips=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("archive", "model", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    save(args.output, evaluate(args.archive, args.model))
