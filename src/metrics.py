import numpy as np

def euclid_err(y_true_xy: np.ndarray, y_pred_xy: np.ndarray) -> np.ndarray:
    d = y_true_xy - y_pred_xy
    return np.sqrt(np.sum(d * d, axis=1))

def error_stats(err_m: np.ndarray):
    return {
        "mean_m": float(np.mean(err_m)),
        "median_m": float(np.median(err_m)),
        "p90_m": float(np.percentile(err_m, 90)),
        "p95_m": float(np.percentile(err_m, 95)),
        "max_m": float(np.max(err_m)),
    }
