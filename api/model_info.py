"""Load the trusted model artifact and describe it for /health and the logs.

The description lets evaluators confirm *which* model is serving: algorithm
family, key hyperparameters, artifact fingerprint and library versions.
"""
import hashlib
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from pathsnap_ml.inference import load

FAMILIES = {
    "HistGradientBoostingClassifier": "Histogram Gradient Boosting",
    "RandomForestClassifier": "Random Forest",
    "DecisionTreeClassifier": "Decision Tree",
    "LogisticRegression": "Logistic Regression",
    "GaussianNB": "Gaussian Naive Bayes",
}
KEY_PARAMETERS = {
    "HistGradientBoostingClassifier": ("learning_rate", "max_iter", "max_leaf_nodes", "max_depth",
                                       "min_samples_leaf", "l2_regularization", "early_stopping"),
    "RandomForestClassifier": ("n_estimators", "max_depth", "min_samples_leaf"),
    "DecisionTreeClassifier": ("max_depth", "min_samples_leaf"),
    "LogisticRegression": ("C", "max_iter"),
    "GaussianNB": ("var_smoothing",),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def final_estimator(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


def _scalar_iterations(estimator):
    # HistGradientBoosting: int boosting rounds. LogisticRegression's n_iter_ is a
    # per-class solver array, not comparable, so it is not reported.
    value = getattr(estimator, "n_iter_", None)
    return int(value) if isinstance(value, (int, np.integer)) else None


def load_with_info(path: Path):
    """Return (artifact, info). Raises like ``pathsnap_ml.inference.load``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifact = load(path)
    estimator = final_estimator(artifact["model"])
    name = type(estimator).__name__
    parameters = estimator.get_params()
    version_warnings = sorted({str(w.message).split(". ")[0] for w in caught
                               if issubclass(w.category, InconsistentVersionWarning)})
    info = dict(
        version=artifact["version"],
        family=FAMILIES.get(name, name),
        estimator=name,
        key_parameters={key: parameters[key] for key in KEY_PARAMETERS.get(name, ())
                        if key in parameters},
        # Boosting rounds actually fitted (early stopping can end before max_iter).
        fitted_iterations=_scalar_iterations(estimator),
        schema=artifact["schema"],
        features=list(artifact["features"]),
        sha256=sha256(path),
        size_bytes=Path(path).stat().st_size,
        sklearn_runtime=sklearn.__version__,
        sklearn_trained=artifact.get("sklearn"),
        version_warnings=version_warnings,
    )
    return artifact, info
