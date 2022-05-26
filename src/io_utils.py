from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from datetime import datetime

def make_run_dir(results_root: str, run_name: str = "") -> Path:
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"run_{ts}" + (f"_{run_name}" if run_name else "")
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _json_default(o):
    """
    Makes json.dump work with numpy types and a few common non-JSON objects.
    """
    # numpy scalars
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)

    # numpy arrays -> lists
    if isinstance(o, np.ndarray):
        return o.tolist()

    # bytes -> str
    if isinstance(o, (bytes, bytearray)):
        try:
            return o.decode("utf-8", errors="replace")
        except Exception:
            return str(o)

    # Path -> str
    if isinstance(o, Path):
        return str(o)

    # fallback
    return str(o)


def save_json(path: Path, obj: dict):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_npz_meta(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"]
    Y = d["Y"]
    train_pts = d["train_pts"].astype(np.int64)
    test_pts = d["test_pts"].astype(np.int64)

    meta = {}
    if "meta" in d.files:
        m = d["meta"]
        if isinstance(m, dict):
            meta = m
        elif isinstance(m, np.ndarray):
            if m.shape == ():
                meta = m.item()
            elif m.size >= 1:
                meta0 = m.flat[0]
                meta = meta0 if isinstance(meta0, dict) else meta0.item()
        else:
            try:
                meta = m.item()
            except Exception:
                meta = {}

    return X, Y, train_pts, test_pts, meta
