import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .review1 import read, save, compare


def run(folder):
    comparison = compare(folder)
    summary = {}
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for row, name in enumerate(("logistic", "naive_bayes")):
        if comparison[name]["status"] != "measured":
            raise ValueError(f"Missing diagnostic result: {name}")
        folds = read(folder / f"{name}.json")["folds"]
        matrix = np.sum([fold["metrics"]["confusion"] for fold in folds], axis=0)
        axes[row, 0].imshow(matrix, cmap="Blues")
        for index in np.ndindex(matrix.shape):
            axes[row, 0].text(index[1], index[0], str(matrix[index]), ha="center", va="center",
                              color="white" if matrix[index] > matrix.max()/2 else "black")
        axes[row, 0].set(title=name, xlabel="Predicted class", ylabel="Proxy class",
                         xticks=[0, 1], yticks=[0, 1])
        trip = sorted(record["top1"] for fold in folds for record in fold["per_trip"].values()
                      if record["top1"] is not None)
        axes[row, 1].plot(trip)
        axes[row, 1].set(xlabel="Trip rank", ylabel="Top-1", ylim=(0, 1))
        conditions = {}
        for key in folds[0]["conditions"]:
            records = [fold["conditions"][key] for fold in folds]
            total = sum(record["observations"] for record in records)
            accepted = sum(record["confident"] for record in records)
            correct = sum(round((record["top1"] or 0)*record["confident"]) for record in records)
            conditions[key] = dict(observations=total, accepted=accepted,
                                   coverage=accepted/total if total else None,
                                   top1=correct/accepted if accepted else None)
        keys = list(conditions)
        for offset, metric in [(-.18, "coverage"), (.18, "top1")]:
            axes[row, 2].barh(np.arange(len(keys))+offset,
                             [conditions[key][metric] or 0 for key in keys], .35, label=metric)
        axes[row, 2].set(yticks=np.arange(len(keys)), yticklabels=keys, xlim=(0, 1))
        axes[row, 2].legend()
        summary[name] = dict(confusion=matrix.tolist(), conditions=conditions,
                             fold_top1=[fold["metrics"]["top1"] for fold in folds],
                             trip_top1_quantiles=np.quantile(trip, [0, .25, .5, .75, 1]).tolist())
    save(folder / "diagnostics.json", summary)
    figure.savefig(folder / "diagnostics.png", dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    run(parser.parse_args().folder)
