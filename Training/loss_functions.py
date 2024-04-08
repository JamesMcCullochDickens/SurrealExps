import torch.nn as nn
import torch


def get_ce_loss(label_smoothing: float = 0.0, ignore_index: int = -100,
                ce_weights=None) -> nn.Module:
    if ce_weights is None:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing,
                                   ignore_index=ignore_index,
                                   weight=ce_weights)
    else:
        return weighted_ce_loss(label_smoothing=label_smoothing,
                                   ignore_index=ignore_index,
                                   ce_weights=ce_weights)


class weighted_ce_loss(nn.Module):
    def __init__(self, ce_weights: torch.Tensor, label_smoothing: float = 0.0, ignore_index: int = -100):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing,
                                           ignore_index=ignore_index,
                                           weight=ce_weights)

    def forward(self, pred: torch.Tensor, gt:torch.Tensor) -> torch.Tensor:
        if self.ce_loss.weight is not None:
            self.ce_loss.weight = self.ce_loss.weight.to(pred.get_device())
        return self.ce_loss(pred, gt)


def soft_dice_coeff(inp: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    num_classes = inp.size(1)
    inp = torch.softmax(inp, dim=1).view(-1, num_classes)
    target = target.view(-1, num_classes)
    intersection = (inp * target).sum(dim=1)
    denominator = inp.sum(dim=1) + target.sum(dim=1)
    dice = (2. * intersection + smooth) / (denominator + smooth)
    return dice.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1 - soft_dice_coeff(inp, target, self.smooth)


def get_dice_loss(smooth: float = 1.0):
    return DiceLoss(smooth)