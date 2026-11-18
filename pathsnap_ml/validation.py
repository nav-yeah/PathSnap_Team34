import math
import re

from .data import SCHEMA
from .review1 import METRICS, PROTOCOL, SEED


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_manifest(manifest):
    require(all(manifest.get(key) == value for key, value in
                dict(protocol=PROTOCOL, schema=SCHEMA, metrics=METRICS, seed=SEED).items()),
            "Incompatible manifest protocol, schema, metric definitions or seed")
    require(bool(re.fullmatch("[a-f0-9]{64}", manifest.get("archive_sha256", ""))),
            "Invalid archive fingerprint")
    folds = manifest.get("validation_trips", [])
    require(len(folds) == 3 and all(folds), "Expected three nonempty folds")
    trips = [trip for fold in folds for trip in fold]
    require(len(trips) == len(set(trips)) == 80, "Expected 80 distinct validation trips")
    return manifest


def validate_metrics(metrics):
    count, accepted = metrics["observations"], metrics["confident"]
    require(type(count) is int and type(accepted) is int and 0 <= accepted <= count,
            "Invalid observation counts")
    require(math.isclose(metrics["coverage"], accepted / count if count else 0),
            "Invalid coverage")
    if not accepted:
        return [0, 0, 0, 0]
    for key in ("accuracy", "precision", "recall", "macro_f1", "weighted_f1", "top1",
                "candidate_recall", "miss_rate", "coverage", "brier"):
        require(0 <= metrics[key] <= 1, f"Invalid {key}")
    require(math.isfinite(metrics["log_loss"]) and metrics["log_loss"] >= 0,
            "Invalid log loss")
    (tn, fp), (fn, tp) = metrics["confusion"]
    cells = [tn, fp, fn, tp]
    require(all(type(value) is int and value >= 0 for value in cells)
            and sum(cells) == 8 * accepted, "Invalid confusion matrix")
    divide = lambda numerator, denominator: numerator / denominator if denominator else 0
    fneg, fpos = divide(2*tn, 2*tn+fp+fn), divide(2*tp, 2*tp+fp+fn)
    expected = dict(accuracy=(tn+tp)/sum(cells), precision=divide(tp,tp+fp),
                    recall=divide(tp,tp+fn), macro_f1=(fneg+fpos)/2,
                    weighted_f1=(fneg*(tn+fp)+fpos*(fn+tp))/sum(cells),
                    candidate_recall=(fn+tp)/accepted, miss_rate=1-(fn+tp)/accepted)
    for key, value in expected.items():
        require(math.isclose(metrics[key], value, abs_tol=1e-12), f"Inconsistent {key}")
    top_hits = metrics["top1"] * accepted
    require(math.isclose(top_hits, round(top_hits), abs_tol=1e-8) and
            top_hits <= fn+tp+1e-8, "Inconsistent Top-1 count")
    return cells


def validate_result(record, manifest, name):
    require(record.get("manifest") == manifest and record.get("model") == name and
            record.get("version") == f"{PROTOCOL}-{name}-baseline", "Incompatible result")
    folds = record["folds"]
    require([fold["split_id"] for fold in folds] == ["fold-0", "fold-1", "fold-2"],
            "Invalid fold IDs")
    for fold, trips in zip(folds, manifest["validation_trips"], strict=True):
        per_trip = fold["per_trip"]
        metrics = fold["metrics"]
        require(set(per_trip) == set(trips) and fold["validation_trips"] == len(trips)
                and fold["training_trips"] == 80-len(trips), "Invalid trip membership")
        cells = validate_metrics(metrics)
        trip_cells = [validate_metrics(row) for row in per_trip.values()]
        require(cells == [sum(column) for column in zip(*trip_cells, strict=True)],
                "Confusion counts do not reconcile")
        require(fold["accepted"] == metrics["confident"] and
                fold["excluded"] + fold["accepted"] == metrics["observations"] and
                all(sum(row[key] for row in per_trip.values()) == metrics[key]
                    for key in ("confident", "observations")), "Observation counts do not reconcile")
        require(round(metrics["top1"] * metrics["confident"]) ==
                sum(round((row["top1"] or 0) * row["confident"]) for row in per_trip.values()),
                "Top-1 counts do not reconcile")
        require(all(math.isfinite(value) and value >= 0 for value in
                    (fold["fit_s"], metrics["predict_s"])), "Invalid timing")
    return folds
