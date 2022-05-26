import torch
import torch.nn as nn
import torch.nn.functional as F

class EuclidHuberLoss(nn.Module):
    def __init__(self, beta=0.02, lam=0.9):
        super().__init__()
        self.beta = beta
        self.lam = lam

    def forward(self, pred, target):
        huber = F.smooth_l1_loss(pred, target, beta=self.beta)
        dist = torch.sqrt(((pred - target) ** 2).sum(dim=1) + 1e-9).mean()
        return huber + self.lam * dist
