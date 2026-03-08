"""
Minimal quantitative comparison of GD-Attention vs softmax attention on the
classic Iris dataset.

This script loads the Iris dataset from scikit-learn, scales the features,
applies both GD-Attention and softmax attention using a leave-one-out evaluation
to provide a simple out-of-sample comparison, and computes simple statistics:
classification accuracy, selection consistency and average per-sample runtime.

It then produces a bar-chart figure highlighting the accuracy and runtime
difference between the two attention mechanisms and writes a small CSV table
with the numerical results.  The Iris dataset is deliberately small and
standardised, making it ideal for a quick yet persuasive comparison.

Usage
-----
Run the script directly from the command line:

```
python iris_comparison.py
```

Upon execution, two files will be written into an ``outputs/`` directory:

- ``iris_quantitative_comparison.png``: grouped bar chart visualising
  accuracy and runtime.
- ``iris_comparison_metrics.csv``: CSV table containing the computed metrics.

These assets can be committed directly into a public repository to accompany
other demonstration material.

"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from main import gd_attention, softmax_attention

# Use a non-interactive backend for safety
plt.switch_backend("Agg")


def evaluate_iris_leave_one_out(alpha: float = 0.5, temperature: float = 1.0) -> dict:
    """
    Load the Iris dataset, standardise it and compute evaluation metrics
    comparing GD-Attention to softmax attention using leave-one-out cross-validation.

    Parameters
    ----------
    alpha: float
        Mixture weight for GD-Attention.
    temperature: float
        Temperature parameter for softmax attention.

    Returns
    -------
    metrics: dict
        Dictionary containing accuracy, selection consistency and average
        per-sample runtimes (in milliseconds) for both methods.
    """
    # Load features and labels
    X, y = load_iris(return_X_y=True)
    # Standardise features to zero mean and unit variance.
    X_scaled = StandardScaler().fit_transform(X)
    n_samples = len(X_scaled)

    correct_gd = 0
    correct_soft = 0
    same_selection = 0
    runtime_gd = 0.0
    runtime_soft = 0.0

    for i in range(n_samples):
        # Leave-one-out split
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False

        keys = X_scaled[mask]
        values = keys.copy()
        key_labels = y[mask]

        query = X_scaled[i]
        query_label = y[i]

        # Softmax attention
        t0 = time.perf_counter()
        sm_res = softmax_attention(query, keys, values, temperature=temperature)
        t1 = time.perf_counter()
        runtime_soft += (t1 - t0)

        # Predicted label: label of key with largest weight
        idx_soft = int(np.argmax(sm_res.weights))
        pred_soft = key_labels[idx_soft]
        if pred_soft == query_label:
            correct_soft += 1

        # GD-Attention
        t0 = time.perf_counter()
        gd_res = gd_attention(query, keys, values, alpha=alpha)
        t1 = time.perf_counter()
        runtime_gd += (t1 - t0)

        idx_gd = int(gd_res.selected_index)
        pred_gd = key_labels[idx_gd]
        if pred_gd == query_label:
            correct_gd += 1

        # Consistency
        if idx_gd == idx_soft:
            same_selection += 1

    return {
        "accuracy_gd": correct_gd / n_samples,
        "accuracy_softmax": correct_soft / n_samples,
        "selection_consistency": same_selection / n_samples,
        "avg_runtime_gd_ms": (runtime_gd / n_samples) * 1000.0,
        "avg_runtime_softmax_ms": (runtime_soft / n_samples) * 1000.0,
    }


def plot_metrics(metrics: dict, save_path: Path) -> None:
    """
    Create a grouped bar chart comparing accuracy and average runtime for
    GD-Attention and softmax attention.

    A selection consistency note is added beneath the plot.  The figure is
    saved to the provided ``save_path``.

    Parameters
    ----------
    metrics: dict
        Resulting metrics dictionary from ``evaluate_iris``.
    save_path: Path
        Destination path for the output PNG file.
    """
    labels = ["Softmax", "GD-Attn"]
    accuracies = [metrics["accuracy_softmax"], metrics["accuracy_gd"]]
    runtimes = [
        metrics["avg_runtime_softmax_ms"],
        metrics["avg_runtime_gd_ms"],
    ]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    # Accuracy subplot
    ax[0].bar(labels, accuracies, color=["tab:blue", "tab:green"], alpha=0.8)
    ax[0].set_ylim(0.0, 1.05)
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Leave-one-out classification on Iris")
    for i, v in enumerate(accuracies):
        ax[0].text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=10)
    ax[0].grid(True, axis="y", alpha=0.2)
    # Runtime subplot
    ax[1].bar(labels, runtimes, color=["tab:blue", "tab:green"], alpha=0.8)
    ax[1].set_ylabel("Avg runtime per sample (ms)")
    ax[1].set_title("Reference runtime")
    for i, v in enumerate(runtimes):
        ax[1].text(i, v + max(runtimes) * 0.05, f"{v:.3f}", ha="center", fontsize=10)
    ax[1].grid(True, axis="y", alpha=0.2)
    # Add selection consistency annotation
    fig.suptitle(
        f"Selection consistency: {metrics['selection_consistency']:.2f}",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.0, 1.0, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(metrics: dict, save_path: Path) -> None:
    """
    Write metrics to a simple CSV file with three rows: accuracy, average
    runtime and selection consistency.  Both method values are written in
    separate columns.

    Parameters
    ----------
    metrics: dict
        Metrics dictionary as returned from ``evaluate_iris``.
    save_path: Path
        Destination path for the CSV file.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        f.write("metric,softmax,gd_attention\n")
        f.write(
            f"accuracy,{metrics['accuracy_softmax']:.4f},{metrics['accuracy_gd']:.4f}\n"
        )
        f.write(
            f"avg_runtime_ms,{metrics['avg_runtime_softmax_ms']:.6f},{metrics['avg_runtime_gd_ms']:.6f}\n"
        )
        f.write(
            f"selection_consistency,{metrics['selection_consistency']:.4f},{metrics['selection_consistency']:.4f}\n"
        )


def main() -> None:
    metrics = evaluate_iris_leave_one_out(alpha=0.5, temperature=1.0)
    out_dir = Path("outputs")
    plot_metrics(metrics, out_dir / "iris_quantitative_comparison.png")
    write_csv(metrics, out_dir / "iris_comparison_metrics.csv")
    print("Metrics:\n", metrics)


if __name__ == "__main__":
    main()