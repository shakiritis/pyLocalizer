import numpy as np
import torch
from torch.utils.data import Dataset

class Seq2DDataset(Dataset):
    """
    X input is (N,T,3,4) and we return Conv2d layout (C,H,W)=(3,T,4).
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, mean: np.ndarray, std: np.ndarray):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.mean = mean.astype(np.float32)  # (3,4)
        self.std = std.astype(np.float32)    # (3,4)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = self.X[i]  # (T,3,4)
        x = (x - self.mean[None, :, :]) / (self.std[None, :, :] + 1e-8)
        x = np.transpose(x, (1, 0, 2))  # (3,T,4)
        return torch.from_numpy(x).float(), torch.from_numpy(self.Y[i]).float()
