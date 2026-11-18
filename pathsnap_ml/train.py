import argparse
import hashlib
import json
import time
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import optuna
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .data import FEATURES, SCHEMA, trips
from .inference import load, predict

SEED = 34


def make_model(name, params=None):
    models = {
        "logistic": (LogisticRegression, dict(max_iter=300, random_state=SEED)),
        "naive_bayes": (GaussianNB, {}),
        "tree": (DecisionTreeClassifier, dict(max_depth=8, random_state=SEED)),
        "forest": (RandomForestClassifier, dict(n_estimators=40, max_depth=10,
                                                n_jobs=2, random_state=SEED)),
        "histogram": (HistGradientBoostingClassifier, dict(max_iter=70, random_state=SEED)),
    }
    estimator, defaults = models[name]
    steps = [SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)]
    if name in {"logistic", "naive_bayes"}:
        steps.append(StandardScaler())
    return make_pipeline(*steps, estimator(**(defaults | (params or {}))))


def assemble(items):
    features, targets, groups, trusted, hits = [], [], [], [], []
    for trip, values, arcs, reference in items:
        if values.shape[:2] != arcs.shape or len(reference) != len(arcs):
            raise ValueError("Invalid trip shapes")
        if arcs.shape[1] != 8:
            raise ValueError("University adapter expects eight candidates")
        labels = arcs == reference[:, None]
        features.append(values)
        targets.append(labels)
        groups.extend([trip] * len(arcs))
        trusted.extend(reference >= 0)
        hits.extend(labels.any(axis=1))
    return tuple(map(np.asarray, (np.concatenate(features), np.concatenate(targets),
                                  groups, trusted, hits)))


def evaluate(model, data, indices, *, by_trip=None, by_condition=None):
    values, labels, _, trusted, hits = data
    start = time.perf_counter()
    scores = model.predict_proba(values[indices].reshape(-1, len(FEATURES)))[:, 1]
    elapsed = time.perf_counter() - start
    scores = scores.reshape(labels[indices].shape)
    if by_trip is not None:
        groups = data[2][indices]
        for trip in sorted(set(groups)):
            mask = groups == trip
            by_trip[trip] = score_metrics(data, indices[mask], scores[mask])
    if by_condition is not None:
        nearest = values[indices, 0, 0]
        speed = values[indices, 0, 4]
        for key, mask in {
            "nearest_under_10m": nearest < 10,
            "nearest_10_to_30m": (nearest >= 10) & (nearest < 30),
            "nearest_30m_or_more": nearest >= 30,
            "speed_missing": ~np.isfinite(speed),
            "speed_under_2mps": speed < 2,
            "speed_2_to_15mps": (speed >= 2) & (speed < 15),
            "speed_15mps_or_more": speed >= 15,
        }.items():
            by_condition[key] = score_metrics(data, indices[mask], scores[mask])
    return score_metrics(data, indices, scores) | {"predict_s": elapsed}


def score_metrics(data, indices, scores):
    _, labels, _, trusted, hits = data
    use = trusted[indices]
    if not use.any():
        return dict(observations=len(indices), confident=0, coverage=0.0,
                    top1=None, candidate_recall=None, miss_rate=None)
    targets, probabilities = labels[indices][use].ravel(), scores[use].ravel()
    predicted = probabilities >= .5
    top = labels[indices][np.arange(len(indices)), scores.argmax(axis=1)]
    return dict(accuracy=accuracy_score(targets, predicted),
                precision=precision_score(targets, predicted, zero_division=0),
                recall=recall_score(targets, predicted, zero_division=0),
                macro_f1=f1_score(targets, predicted, labels=[0, 1], average="macro", zero_division=0),
                weighted_f1=f1_score(targets, predicted, average="weighted"),
                log_loss=log_loss(targets, probabilities, labels=[0, 1]),
                brier=brier_score_loss(targets, probabilities),
                confusion=confusion_matrix(targets, predicted, labels=[0, 1]).tolist(),
                top1=float(top[use].mean()), coverage=float(use.mean()),
                candidate_recall=float(hits[indices][use].mean()),
                miss_rate=float(1 - hits[indices][use].mean()),
                observations=len(indices), confident=int(use.sum()))


def fit(name, params, data, indices):
    values, labels, _, trusted, _ = data
    indices = indices[trusted[indices]]
    model = make_model(name, params)
    start = time.perf_counter()
    model.fit(values[indices].reshape(-1, len(FEATURES)), labels[indices].ravel())
    return model, time.perf_counter() - start


def search_space(name, trial):
    if name == "logistic":
        return {"C": trial.suggest_float("C", .01, 30, log=True)}
    if name == "naive_bayes":
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-11, .1, log=True)}
    if name in {"tree", "forest"}:
        return {"max_depth": trial.suggest_int("max_depth", 3, 14),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 80, log=True)}
    return {"learning_rate": trial.suggest_float("learning_rate", .02, .25, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 40),
            "l2_regularization": trial.suggest_float("l2_regularization", .01, 10, log=True)}


def run(source, output, count):
    output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    write("config.json", dict(seed=SEED, folds=3, trials=count, startup_trials=4,
                             objective="mean fold top1", baseline_tiebreak="mean fold log_loss",
                             archive_sha256=digest, sklearn=sklearn.__version__,
                             optuna=optuna.__version__, numpy=np.__version__,
                             alignment_radius_m=35, alignment_margin_m=8, candidates=8))
    with ZipFile(source) as archive:
        train = assemble(trips(archive, "train"))
    write("train_alignment.json", alignment_summary(train))
    folds = list(GroupKFold(3).split(train[0], groups=train[2]))
    write("folds.json", [sorted(set(train[2][valid])) for _, valid in folds])

    def cv(name, params=None):
        metrics = []
        for train_rows, valid in folds:
            model, seconds = fit(name, params, train, train_rows)
            metrics.append(evaluate(model, train, valid) | {"fit_s": seconds})
        return metrics

    def mean(rows, key):
        return float(np.mean([row[key] for row in rows]))

    baselines = {}
    for name in ("logistic", "naive_bayes", "tree", "forest", "histogram"):
        model = make_model(name)
        baselines[name] = {"folds": cv(name), "parameters": model.steps[-1][1].get_params(),
                           "pipeline": repr(model)}
        write("baselines.json", baselines)
        print(name, mean(baselines[name]["folds"], "top1"), flush=True)
    selected = min(baselines, key=lambda name: (-mean(baselines[name]["folds"], "top1"),
                                              mean(baselines[name]["folds"], "log_loss")))
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=4))

    def objective(trial):
        rows = cv(selected, search_space(selected, trial))
        trial.set_user_attr("folds", rows)
        return mean(rows, "top1")

    def persist(_, trial):
        with (output / "trials.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(number=trial.number, state=trial.state.name,
                                         value=trial.value, params=trial.params,
                                         seconds=trial.duration.total_seconds(),
                                         **trial.user_attrs)) + "\n")

    study.optimize(objective, n_trials=count, callbacks=[persist])
    baseline = mean(baselines[selected]["folds"], "top1")
    use_tuned = study.best_value > baseline
    write("selection.json", dict(model=selected, baseline_cv_top1=baseline,
                                 best_cv_top1=study.best_value, use_tuned=use_tuned,
                                 best_parameters=study.best_params,
                                 best_trial=study.best_trial.number))
    with ZipFile(source) as archive:
        test = assemble(trips(archive, "student_test"))
    write("student_alignment.json", alignment_summary(test))
    if set(train[2]) & set(test[2]):
        raise ValueError("Overlapping trip identifiers")
    results = {}
    for label, name, params in [(name, name, None) for name in baselines] + [
        ("tuned", selected, study.best_params)
    ]:
        model, seconds = fit(name, params, train, np.arange(len(train[0])))
        results[label] = evaluate(model, test, np.arange(len(test[0]))) | {"fit_s": seconds}
        write("student_test.json", results)
        if label == ("tuned" if use_tuned else selected):
            metadata = dict(schema=SCHEMA, features=FEATURES,
                            version=f"{selected}-{digest[:10]}-{study.best_trial.number if use_tuned else 'baseline'}",
                            parameters=model.steps[-1][1].get_params(),
                            sklearn=sklearn.__version__)
            write("model.json", metadata)
            artifact_path = output / "model.joblib"
            joblib.dump(metadata | {"model": model}, artifact_path, compress=3)
            sample = test[0][0]
            rows = [dict(trip_id="smoke", observation_index=0, timestamp=0, arc_id=index,
                         features=dict(zip(FEATURES, vector, strict=True)))
                    for index, vector in enumerate(sample)]
            scored = predict(load(artifact_path), rows)
            np.testing.assert_allclose([row["score"] for row in scored],
                                       model.predict_proba(sample)[:, 1])
            write("artifact_verification.json", dict(reloaded=True, reference_free_rows=len(scored)))


def alignment_summary(data):
    _, _, trips, trusted, hits = data
    records = {}
    for trip in sorted(set(trips)):
        indices = trips == trip
        records[trip] = dict(observations=int(indices.sum()), confident=int(trusted[indices].sum()),
                             uncertain=int((~trusted[indices]).sum()),
                             candidate_misses=int((trusted & ~hits & indices).sum()))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()
    if args.trials < 5:
        parser.error("At least five trials are needed to go beyond TPE startup")
    with threadpool_limits(limits=2):
        run(args.archive, args.output, args.trials)
