import joblib
import numpy as np

from .data import FEATURES, SCHEMA


def matrix(records):
    values = np.empty((len(records), len(FEATURES)))
    for index, record in enumerate(records):
        if set(record) != set(FEATURES):
            raise ValueError("Feature keys do not match schema")
        values[index] = [np.nan if record[key] is None else record[key] for key in FEATURES]
    values[~np.isfinite(values)] = np.nan
    return values


def load(path):
    artifact = joblib.load(path)
    if artifact["schema"] != SCHEMA or tuple(artifact["features"]) != FEATURES:
        raise ValueError("Incompatible artifact schema")
    return artifact


def predict(artifact, rows):
    rows = list(rows)
    keys = ("trip_id", "observation_index", "timestamp", "arc_id")
    seen, timestamps = set(), {}
    for row in rows:
        trip, index, time, arc = (row[key] for key in keys)
        if (trip, index, arc) in seen or not np.isfinite(time):
            raise ValueError("Duplicate candidate or invalid timestamp")
        seen.add((trip, index, arc))
        if timestamps.setdefault((trip, index), time) != time:
            raise ValueError("Inconsistent observation timestamp")
    if not rows:
        return []
    scores = artifact["model"].predict_proba(matrix([row["features"] for row in rows]))[:, 1]
    return [dict({key: row[key] for key in keys}, score=float(score),
                 schema=SCHEMA, model_version=artifact["version"])
            for row, score in zip(rows, scores, strict=True)]
