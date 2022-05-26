from __future__ import annotations
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def plot_train_val_curves(curves_csv: Path, out_png: Path):
    import csv
    rows = []
    with Path(curves_csv).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    ep = np.array([int(x["epoch"]) for x in rows])
    tr = np.array([float(x["train_loss"]) for x in rows])

    mean_m = np.array([float(x["mean_m"]) for x in rows])
    med_m  = np.array([float(x["median_m"]) for x in rows])
    p95_m  = np.array([float(x["p95_m"]) for x in rows])

    # Plot 1: train loss
    plt.figure(figsize=(8, 5))
    plt.plot(ep, tr)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss vs Epoch")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_png.with_name("train_loss.png"), dpi=200)
    plt.close()

    # Plot 2: validation stats
    plt.figure(figsize=(8, 5))
    plt.plot(ep, mean_m, label="val mean (m)")
    plt.plot(ep, med_m, label="val median (m)")
    plt.plot(ep, p95_m, label="val p95 (m)")
    plt.xlabel("Epoch")
    plt.ylabel("Error (m)")
    plt.title("Validation Error vs Epoch")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_error_cdf(err: np.ndarray, out_png: Path):
    err = np.asarray(err).astype(np.float32)
    xs = np.sort(err)
    ys = np.arange(1, xs.size + 1) / xs.size

    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys)
    plt.xlabel("Euclidean error (m)")
    plt.ylabel("CDF")
    plt.title("Test Error CDF")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
