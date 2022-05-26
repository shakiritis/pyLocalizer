from __future__ import annotations
import math
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import Seq2DDataset
from src.model import CNN2D_TimeAnchor
from src.losses import EuclidHuberLoss
from src.metrics import euclid_err, error_stats
from src.augmentation import delaunay_aug_phase_correct

def lr_for_epoch(ep, base_lr, total_epochs):
    t = ep / max(1, total_epochs - 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))

def train_one(model, loader, opt, loss_fn, device, ep, base_lr, total_epochs):
    model.train()
    for pg in opt.param_groups:
        pg["lr"] = lr_for_epoch(ep, base_lr, total_epochs)

    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(1, n)

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        preds.append(pred)
        trues.append(yb.numpy())
    yp = np.vstack(preds).astype(np.float32)
    yt = np.vstack(trues).astype(np.float32)
    err = euclid_err(yt, yp)
    return error_stats(err), err

def run_training(
    X: np.ndarray,
    Y: np.ndarray,
    train_pts: np.ndarray,
    test_pts: np.ndarray,
    run_dir: Path,
    train_cfg: dict,
    aug_cfg: dict,
    select_metric: str = "p95_m",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    EPOCHS = int(train_cfg["epochs"])
    BATCH = int(train_cfg["batch"])
    LR = float(train_cfg["lr"])
    WEIGHT_DECAY = float(train_cfg["weight_decay"])
    HUBER_BETA = float(train_cfg["huber_beta"])
    LAM_EUCLID = float(train_cfg["lam_euclid"])

    # Base splits
    Xtr = X[train_pts]
    ytr = Y[train_pts]
    Xte = X[test_pts]
    yte = Y[test_pts]

    # Optional Delaunay augmentation (done once, deterministic)
    if aug_cfg.get("use_delaunay_aug", True):
        rng = np.random.default_rng(int(aug_cfg["aug_seed"]))
        Xaug, yaug = delaunay_aug_phase_correct(
            X=X, Y=Y, train_pts=train_pts,
            aug_per_simplex=int(aug_cfg["aug_per_simplex"]),
            max_simplices=int(aug_cfg["max_simplices"]),
            rng=rng
        )
        if Xaug.shape[0] > 0:
            Xtr = np.concatenate([Xtr, Xaug], axis=0)
            ytr = np.concatenate([ytr, yaug], axis=0)

    # Normalize using train only
    flat = Xtr.reshape(-1, 3, 4)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)

    train_ds = Seq2DDataset(Xtr, ytr, mean, std)
    test_ds = Seq2DDataset(Xte, yte, mean, std)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, drop_last=False)

    model = CNN2D_TimeAnchor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = EuclidHuberLoss(beta=HUBER_BETA, lam=LAM_EUCLID)

    curves = []
    best_val = float("inf")
    best_state = None

    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one(model, train_loader, opt, loss_fn, device, ep - 1, LR, EPOCHS)
        stats, _ = eval_model(model, test_loader, device)

        curves.append({
            "epoch": ep,
            "train_loss": tr_loss,
            **stats,
        })

        val_key = float(stats[select_metric])
        if val_key < best_val:
            best_val = val_key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 5 == 0:
            print(f"[EP {ep:03d}] loss={tr_loss:.5f} val_mean={stats['mean_m']:.4f} val_p95={stats['p95_m']:.4f}")

    # Save last checkpoint
    last_ckpt = {
        "state_dict": model.state_dict(),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "train_pts": train_pts,
        "test_pts": test_pts,
        "train_cfg": train_cfg,
        "aug_cfg": aug_cfg,
        "select_metric": select_metric,
        "device": device,
    }
    torch.save(last_ckpt, run_dir / "final_ckpt.pt")

    # Load best and save best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    best_ckpt = dict(last_ckpt)
    best_ckpt["state_dict"] = model.state_dict()
    best_ckpt["best_val"] = best_val
    torch.save(best_ckpt, run_dir / "best_ckpt.pt")

    # Final eval + save errors array
    final_stats, err = eval_model(model, test_loader, device)
    np.save(run_dir / "errors_test.npy", err.astype(np.float32))

    print("\nTEST_ERROR (best):")
    for k, v in final_stats.items():
        print(f"  {k}: {v:.6f}")

    # Save curves CSV
    save_curves_csv(run_dir / "curves.csv", curves)

    # Plot curves + CDF
    from src.plotting import plot_train_val_curves, plot_error_cdf
    plot_train_val_curves(run_dir / "curves.csv", run_dir / "train_val_curves.png")
    plot_error_cdf(err, run_dir / "error_cdf.png")

def save_curves_csv(path: Path, curves: list[dict]):
    import csv
    fieldnames = list(curves[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in curves:
            w.writerow(row)
