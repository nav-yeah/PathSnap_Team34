import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(folder):
    baseline = json.loads((folder / "baselines.json").read_text(encoding="utf-8"))
    heldout = json.loads((folder / "student_test.json").read_text(encoding="utf-8"))
    trials = [json.loads(line) for line in (folder / "trials.jsonl").read_text().splitlines()]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axes[0, 0].bar(list(baseline), [np.mean([fold["macro_f1"] for fold in record["folds"]])
                                    for record in baseline.values()])
    axes[0, 0].set_title("Grouped CV: candidate macro F1")
    positions = np.arange(len(heldout))
    for offset, metric in [(-.2, "top1"), (.2, "macro_f1")]:
        axes[0, 1].bar(positions + offset, [row[metric] for row in heldout.values()],
                       .4, label=metric)
    axes[0, 1].set_xticks(positions, list(heldout), rotation=25)
    axes[0, 1].set_title("Student test")
    axes[0, 1].legend()
    axes[1, 0].bar(list(heldout), [row["log_loss"] for row in heldout.values()])
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].set_title("Student test: candidate log loss")
    axes[1, 1].plot([row["number"] for row in trials], [row["value"] for row in trials], "o-")
    axes[1, 1].set(xlabel="Trial", ylabel="Mean CV Top-1", title="Optuna TPE")
    figure.savefig(folder / "comparison.png", dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    plot(parser.parse_args().folder)
