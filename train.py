#!/usr/bin/env python3
import argparse
from src.io_utils import make_run_dir, save_json
from src.utils_seed import seed_everything
from src.io_utils import load_npz_meta
from src.train_eval import run_training

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to dataset .npz")
    ap.add_argument("--results_root", type=str, default="results", help="Root folder for runs")
    ap.add_argument("--run_name", type=str, default="", help="Optional run name suffix")

    # Train hparams
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--huber_beta", type=float, default=0.02)
    ap.add_argument("--lam_euclid", type=float, default=0.9)

    # Aug hparams
    ap.add_argument("--use_delaunay_aug", action="store_true", default=True)
    ap.add_argument("--aug_seed", type=int, default=12345)
    ap.add_argument("--aug_per_simplex", type=int, default=12)
    ap.add_argument("--max_simplices", type=int, default=1500)

    # Selection metric
    ap.add_argument("--select_metric", type=str, default="p95_m", choices=["mean_m","median_m","p90_m","p95_m","max_m"])

    return ap

def main():
    args = build_argparser().parse_args()

    X, Y, train_pts, test_pts, meta = load_npz_meta(args.data)
    seed = int(meta.get("seed", 7))
    seed_everything(seed)

    run_dir = make_run_dir(args.results_root, args.run_name)

    # Save run config immediately
    run_config = {
        "data": args.data,
        "seed": seed,
        "train_hparams": {
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "huber_beta": args.huber_beta,
            "lam_euclid": args.lam_euclid,
        },
        "aug_hparams": {
            "use_delaunay_aug": bool(args.use_delaunay_aug),
            "aug_seed": args.aug_seed,
            "aug_per_simplex": args.aug_per_simplex,
            "max_simplices": args.max_simplices,
        },
        "select_metric": args.select_metric,
        "meta_from_data": meta,
    }
    save_json(run_dir / "run_meta.json", run_config)

    run_training(
        X=X, Y=Y,
        train_pts=train_pts, test_pts=test_pts,
        run_dir=run_dir,
        train_cfg=run_config["train_hparams"],
        aug_cfg=run_config["aug_hparams"],
        select_metric=args.select_metric,
    )

    print(f"\n[DONE] Results saved in: {run_dir}")

if __name__ == "__main__":
    main()
