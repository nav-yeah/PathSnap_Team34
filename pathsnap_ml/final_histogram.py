"""Nested training-only selection followed by one frozen student evaluation."""
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import optuna
import sklearn
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .data import FEATURES, SCHEMA, trips
from .inference import load
from .review1 import save
from .train import assemble, evaluate, fit, search_space, SEED


def splits(data, indices, count):
    groups = data[2][indices]
    for train, valid in GroupKFold(count).split(indices, groups=groups):
        yield indices[train], indices[valid]


def run(source, output):
    output.mkdir(parents=True)
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != "4804b0399d1d56f4844b9c74e2f998aae0a281fb5947a1f47ad1bd013214d248":
        raise ValueError("Unexpected university archive")
    with ZipFile(source) as archive:
        data = assemble(trips(archive, "train"))
    groups = data[2]
    indices = np.arange(len(groups))
    save(output / "protocol.json", dict(archive_sha256=digest, seed=SEED, outer=3,
         inner=2, trials_per_study=5, startup_trials=2, sampler="TPE",
         objective="mean accepted-observation Top-1 including candidate misses",
         search_space=dict(learning_rate=[.02,.25,"log"], max_leaf_nodes=[8,40],
                           l2_regularization=[.01,10,"log"])))

    def tune(rows, name):
        folds = list(splits(data, rows, 2))
        save(output / f"{name}-splits.json", [sorted(set(groups[valid])) for _, valid in folds])
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
            seed=SEED, n_startup_trials=2))

        def objective(trial):
            params = search_space("histogram", trial)
            scores = []
            for train, valid in folds:
                model, _ = fit("histogram", params, data, train)
                scores.append(evaluate(model, data, valid)["top1"])
            return float(np.mean(scores))

        def checkpoint(study, trial):
            save(output / f"{name}-trials.json", [
                dict(number=item.number, params=item.params, value=item.value,
                     state=item.state.name, seconds=item.duration.total_seconds())
                for item in study.trials])

        study.optimize(objective, n_trials=5, callbacks=[checkpoint])
        return study.best_params, study.best_value

    outer = []
    for number, (train, valid) in enumerate(splits(data, indices, 3)):
        params, score = tune(train, f"outer-{number}")
        record = dict(validation_trips=sorted(set(groups[valid])), params=params, inner_top1=score)
        for name, configuration in [("baseline", None), ("tuned", params)]:
            model, seconds = fit("histogram", configuration, data, train)
            per_trip = {}
            record[name] = evaluate(model, data, valid, by_trip=per_trip) | dict(
                fit_s=seconds, per_trip=per_trip)
        outer.append(record)
        save(output / "outer.json", outer)
    use_tuned = sum(row["tuned"]["top1"] for row in outer) > sum(
        row["baseline"]["top1"] for row in outer)
    params, score = tune(indices, "final") if use_tuned else ({}, None)
    model, seconds = fit("histogram", params, data, indices)
    metadata = dict(schema=SCHEMA, features=FEATURES, version="histogram-final-v1",
                    archive_sha256=digest, tuned=use_tuned, inner_top1=score,
                    parameters=model.steps[-1][1].get_params(), fit_s=seconds,
                    trained_utc=datetime.now(timezone.utc).isoformat(),
                    sklearn=sklearn.__version__, numpy=np.__version__, optuna=optuna.__version__)
    joblib.dump(metadata | {"model": model}, output / "model.joblib", compress=3)
    restored = load(output / "model.joblib")
    sample = data[0][0]
    predictions = restored["model"].predict_proba(sample)
    np.testing.assert_allclose(predictions, model.predict_proba(sample))
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite reference-free scores")
    save(output / "model.json", metadata | {"reference_free_reload_verified": True})
    with ZipFile(source) as archive:
        test = assemble(trips(archive, "student_test"))
    if set(test[2]) & set(groups):
        raise ValueError("Partition overlap")
    per_trip = {}
    metrics = evaluate(restored["model"], test, np.arange(len(test[0])), by_trip=per_trip)
    save(output / "student.json", dict(partition="student_test", metrics=metrics, per_trip=per_trip))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.archive, args.output)
