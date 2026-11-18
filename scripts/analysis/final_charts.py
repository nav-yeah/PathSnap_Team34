"""Plot recorded tuning and frozen route metrics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(tuning, routes, output):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for path in sorted(tuning.glob("outer-*-trials.json")):
        rows = json.loads(path.read_text())
        axes[0].plot([row["number"] for row in rows],
                     [100*row["value"] for row in rows], "o-", label=path.stem)
    axes[0].set(xlabel="TPE trial", ylabel="Inner Top-1 (%)")
    axes[0].legend()
    totals = json.loads(routes.read_text())["totals"]
    axes[1].bar(list(totals), [100*row["top1"] for row in totals.values()])
    axes[1].set(title="Frozen student proxy Top-1", ylabel="Percent", ylim=(0, 100))
    figure.savefig(output, dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("tuning", "routes", "output"):
        parser.add_argument(name, type=Path)
    plot(**vars(parser.parse_args()))
