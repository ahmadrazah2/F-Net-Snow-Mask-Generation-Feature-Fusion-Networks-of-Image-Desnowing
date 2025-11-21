import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, eps=1e-7):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        BCE_loss = F.binary_cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        focal_loss = focal_loss.mean()

        intersection = (preds * targets).sum()
        dice_score = (2. * intersection + self.eps) / (preds.sum() + targets.sum() + self.eps)
        dice_loss = 1 - dice_score

        total_loss = focal_loss * (1 + dice_loss)
        return total_loss
