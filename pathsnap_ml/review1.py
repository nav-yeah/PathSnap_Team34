"""Untuned, training-only team comparison."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .data import FEATURES, SCHEMA, trips

MODELS = ("logistic", "naive_bayes", "tree", "forest", "histogram")
PROTOCOL = "review1-v1"
SEED = 34
METRICS = "review1-v1:binary-threshold-0.5;top1-accepted-including-misses;equal-fold-mean"


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(archive_path, output, names):
    import joblib
    import sklearn
    from sklearn.model_selection import GroupKFold
    from threadpoolctl import threadpool_limits

    from .inference import load, predict
    from .train import assemble, evaluate, fit

    output.mkdir(parents=True, exist_ok=True)
    with archive_path.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    with ZipFile(archive_path) as archive:
        data = assemble(trips(archive, "train"))
    groups = data[2]
    folds = list(GroupKFold(3).split(data[0], groups=groups))
    manifest = dict(protocol=PROTOCOL, schema=SCHEMA, metrics=METRICS, archive_sha256=archive_hash,
                    seed=SEED, sklearn=sklearn.__version__, numpy=np.__version__,
                    validation_trips=[sorted(set(groups[valid])) for _, valid in folds])
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError("Output already contains an incompatible split manifest")
    save(manifest_path, manifest)
    with threadpool_limits(limits=2):
        for name in names:
            version = f"{PROTOCOL}-{name}-baseline"
            records = []
            for number, (training, validation) in enumerate(folds):
                if set(groups[training]) & set(groups[validation]):
                    raise ValueError("Trip leakage")
                model, fit_s = fit(name, None, data, training)
                per_trip = {}
                conditions = {}
                metrics = evaluate(model, data, validation, by_trip=per_trip,
                                   by_condition=conditions)
                records.append(dict(split_id=f"fold-{number}", fit_s=fit_s,
                                    training_trips=len(set(groups[training])),
                                    validation_trips=len(set(groups[validation])),
                                    accepted=metrics["confident"],
                                    excluded=metrics["observations"] - metrics["confident"],
                                    metrics=metrics, per_trip=per_trip, conditions=conditions))
            model, full_fit_s = fit(name, None, data, np.arange(len(groups)))
            artifact_path = output / f"{name}.joblib"
            parameters = model.steps[-1][1].get_params()
            joblib.dump(dict(schema=SCHEMA, features=FEATURES, version=version,
                             parameters=parameters, model=model), artifact_path, compress=3)
            sample = data[0][0]
            rows = [dict(trip_id="smoke", observation_index=0, timestamp=0, arc_id=index,
                         features=dict(zip(FEATURES, vector, strict=True)))
                    for index, vector in enumerate(sample)]
            np.testing.assert_allclose([row["score"] for row in predict(load(artifact_path), rows)],
                                       model.predict_proba(sample)[:, 1])
            save(output / f"{name}.json", dict(model=name, version=version, manifest=manifest,
                                              hyperparameters=parameters, pipeline=repr(model),
                                              artifact_path=artifact_path.name, full_fit_s=full_fit_s,
                                              folds=records, reference_free_reload_verified=True))


def compare(folder, submissions=()):
    from .validation import validate_manifest, validate_result

    results = {}
    manifest = None
    paths = {}
    for directory in (folder, *submissions):
        candidate = directory / "manifest.json"
        if candidate.exists():
            submitted = validate_manifest(read(candidate))
            if manifest is not None and submitted != manifest:
                raise ValueError(f"Incompatible manifest: {candidate}")
            manifest = submitted
        for name in MODELS:
            path = directory / f"{name}.json"
            if path.exists():
                if not candidate.exists() or name in paths:
                    raise ValueError(f"Missing manifest or duplicate submission: {path}")
                paths[name] = path
    for name in MODELS:
        if name not in paths:
            results[name] = {"status": "pending"}
            continue
        folds = validate_result(read(paths[name]), manifest, name)
        results[name] = {"status": "measured", **{
            metric: float(np.mean([fold["metrics"][metric] for fold in folds]))
            for metric in ("accuracy", "precision", "recall", "macro_f1", "top1",
                           "candidate_recall", "miss_rate", "coverage", "predict_s")
        }, "top1_fold_sd": float(np.std([fold["metrics"]["top1"] for fold in folds])),
                        "mean_fit_s": float(np.mean([fold["fit_s"] for fold in folds]))}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("archive", type=Path)
    training.add_argument("output", type=Path)
    training.add_argument("--models", nargs="+", choices=MODELS,
                          default=["logistic", "naive_bayes"])
    comparison = commands.add_parser("compare")
    comparison.add_argument("folder", type=Path)
    comparison.add_argument("submissions", nargs="*", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        run(args.archive, args.output, args.models)
    else:
        print(json.dumps(compare(args.folder, args.submissions), indent=2))
