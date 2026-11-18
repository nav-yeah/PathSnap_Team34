"""Install the team's final model artifact for the API, after checking it.

    python deploy/promote_model.py <source.joblib> [--dest artifacts/model.joblib]
                                   [--expect-family "Histogram Gradient Boosting"]

Checks: schema and feature contract (pathsnap_ml.inference.load), expected
algorithm family, scikit-learn version compatibility, and that it scores a
reference-free candidate row through the same predict() the API uses.
Never pass a file from an untrusted source: joblib files can execute code.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.model_info import load_with_info, sha256  # noqa: E402
from pathsnap_ml.data import FEATURES  # noqa: E402
from pathsnap_ml.inference import predict  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--dest", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--expect-family", default="Histogram Gradient Boosting")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    args = parser.parse_args()

    artifact, info = load_with_info(args.source)
    problems = []
    if args.expect_family and info["family"] != args.expect_family:
        problems.append(f"family is {info['family']!r}, expected {args.expect_family!r}")
    if info["version_warnings"] and not args.allow_version_mismatch:
        problems.append("scikit-learn version differs from training: " + "; ".join(info["version_warnings"]))
    row = dict(trip_id="promote-check", observation_index=0, timestamp=0.0, arc_id=0,
               features=dict(zip(FEATURES, (5.0, 0.9, 0.5, 40.0, 10.0), strict=True)))
    score = predict(artifact, [row])[0]["score"]
    if not 0.0 <= score <= 1.0:
        problems.append(f"score {score} outside [0, 1]")
    print(json.dumps({k: v for k, v in info.items() if k != "features"}, indent=2, default=str))
    if problems:
        print("REFUSED:\n  - " + "\n  - ".join(problems))
        return 1
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    if args.source.resolve() != args.dest.resolve():
        shutil.copyfile(args.source, args.dest)
    assert sha256(args.dest) == info["sha256"]
    print(f"OK: installed {info['family']} ({info['version']}) at {args.dest}, sha256 {info['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
