import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathsnap_ml.review1 import compare, read, save


def generate(output, baseline, trees, verified, legacy):
    results = compare(baseline, (trees, verified))
    if any(row["status"] != "measured" for row in results.values()):
        raise ValueError("Expected five validated models")
    save(output / "comparison.json", results)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", *next(iter(results.values()))])
        writer.writeheader()
        writer.writerows(dict(model=name, **row) for name, row in results.items())
    forest = read(trees / "forest.json")
    histogram = read(verified / "histogram.json")
    original = read(legacy / "histogram.json")
    differences = []
    for left, right in zip(forest["folds"], histogram["folds"], strict=True):
        for trip, record in left["per_trip"].items():
            other = right["per_trip"][trip]
            if record["confident"] != other["confident"]:
                raise ValueError("Accepted counts differ")
            if record["confident"]:
                differences.append(record["top1"] - other["top1"])
    paired = np.asarray(differences)
    rng = np.random.default_rng(34)
    bootstrap = rng.choice(paired, (10000, len(paired)), replace=True).mean(axis=1)
    save(output / "validation.json", dict(
        manifest=histogram["manifest"], all_five_validated=True,
        histogram_reload_verified=histogram["reference_free_reload_verified"],
        legacy_histogram_missing_metric_id="metrics" not in original["manifest"],
        legacy_histogram_fold_top1=[fold["metrics"]["top1"] for fold in original["folds"]],
        verified_histogram_fold_top1=[fold["metrics"]["top1"] for fold in histogram["folds"]],
        paired=dict(trips=len(paired), mean_rf_minus_hist=float(paired.mean()),
                    wins=int((paired > 0).sum()), ties=int((paired == 0).sum()),
                    losses=int((paired < 0).sum()),
                    descriptive_trip_bootstrap_95=np.quantile(bootstrap, [.025, .975]).tolist()),
    ))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, metric in zip(axes, ("top1", "macro_f1"), strict=True):
        axis.bar(list(results), [100 * row[metric] for row in results.values()],
                 yerr=[100*row["top1_fold_sd"] for row in results.values()] if metric == "top1" else None)
        axis.set(title=metric, ylabel="Percent", ylim=(0, 100))
        axis.tick_params(axis="x", rotation=25)
    figure.savefig(output / "comparison.png", dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("output", "baseline", "trees", "verified", "legacy"):
        parser.add_argument(name, type=Path)
    generate(**vars(parser.parse_args()))
