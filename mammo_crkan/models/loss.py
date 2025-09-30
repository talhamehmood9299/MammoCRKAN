
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Multi-class focal loss with logits input.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        """
        logits: (B, C), raw logits
        target: (B,), int labels
        """
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce)  # = softmax(logits)[target]
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
