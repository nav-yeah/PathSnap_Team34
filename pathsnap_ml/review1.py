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
    manifest = dict(protocol=PROTOCOL, schema=SCHEMA, archive_sha256=archive_hash,
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
                metrics = evaluate(model, data, validation, by_trip=per_trip)
                records.append(dict(split_id=f"fold-{number}", fit_s=fit_s,
                                    training_trips=len(set(groups[training])),
                                    validation_trips=len(set(groups[validation])),
                                    accepted=metrics["confident"],
                                    excluded=metrics["observations"] - metrics["confident"],
                                    metrics=metrics, per_trip=per_trip))
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


def compare(folder):
    results = {}
    manifest = read(folder / "manifest.json") if (folder / "manifest.json").exists() else None
    for name in MODELS:
        path = folder / f"{name}.json"
        if not path.exists():
            results[name] = {"status": "pending"}
            continue
        record = read(path)
        if manifest is None or record["manifest"] != manifest or record["model"] != name:
            raise ValueError(f"Incompatible result: {path}")
        folds = record["folds"]
        if [fold["split_id"] for fold in folds] != ["fold-0", "fold-1", "fold-2"]:
            raise ValueError(f"Invalid fold IDs: {path}")
        results[name] = {"status": "measured", **{
            metric: float(np.mean([fold["metrics"][metric] for fold in folds]))
            for metric in ("accuracy", "precision", "recall", "macro_f1", "top1",
                           "candidate_recall", "miss_rate", "coverage", "predict_s")
        }, "mean_fit_s": float(np.mean([fold["fit_s"] for fold in folds]))}
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
    args = parser.parse_args()
    if args.command == "train":
        run(args.archive, args.output, args.models)
    else:
        print(json.dumps(compare(args.folder), indent=2))
