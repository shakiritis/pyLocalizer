from __future__ import annotations
import numpy as np
from scipy.spatial import Delaunay

def sample_barycentric(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.dirichlet(np.ones(3), size=n).astype(np.float32)

def delaunay_aug_phase_correct(
    X: np.ndarray,        # (P,T,3,4)
    Y: np.ndarray,        # (P,2)
    train_pts: np.ndarray,
    aug_per_simplex: int,
    max_simplices: int,
    rng: np.random.Generator,
):
    coords = Y[train_pts]  # (Ptr,2)
    if coords.shape[0] < 3:
        return np.zeros((0,) + X.shape[1:], dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    tri = Delaunay(coords)
    simplices = tri.simplices

    if max_simplices > 0 and simplices.shape[0] > max_simplices:
        pick = rng.choice(simplices.shape[0], size=max_simplices, replace=False)
        simplices = simplices[pick]

    X_list, y_list = [], []

    for s in simplices:
        pA = train_pts[int(s[0])]
        pB = train_pts[int(s[1])]
        pC = train_pts[int(s[2])]

        XA, XB, XC = X[pA], X[pB], X[pC]
        yA, yB, yC = Y[pA], Y[pB], Y[pC]

        w = sample_barycentric(aug_per_simplex, rng)
        for i in range(aug_per_simplex):
            w1, w2, w3 = float(w[i, 0]), float(w[i, 1]), float(w[i, 2])

            r_syn = (w1 * XA[:, 0, :] + w2 * XB[:, 0, :] + w3 * XC[:, 0, :]).astype(np.float32)

            uA = XA[:, 1, :].astype(np.float32) + 1j * XA[:, 2, :].astype(np.float32)
            uB = XB[:, 1, :].astype(np.float32) + 1j * XB[:, 2, :].astype(np.float32)
            uC = XC[:, 1, :].astype(np.float32) + 1j * XC[:, 2, :].astype(np.float32)

            u_syn = (w1 * uA + w2 * uB + w3 * uC)
            u_syn = u_syn / (np.abs(u_syn) + 1e-8)

            c_syn = np.real(u_syn).astype(np.float32)
            s_syn = np.imag(u_syn).astype(np.float32)

            Xsyn = np.zeros_like(XA, dtype=np.float32)
            Xsyn[:, 0, :] = r_syn
            Xsyn[:, 1, :] = c_syn
            Xsyn[:, 2, :] = s_syn

            ysyn = (w1 * yA + w2 * yB + w3 * yC).astype(np.float32)

            X_list.append(Xsyn)
            y_list.append(ysyn)

    if not X_list:
        return np.zeros((0,) + X.shape[1:], dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    return np.stack(X_list, axis=0), np.stack(y_list, axis=0)
