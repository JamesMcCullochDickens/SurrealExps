import torch.nn as nn
import torch


def get_ce_loss(label_smoothing: float = 0.0, ignore_index: int = -100,
                ce_weights=None) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing,
                               ignore_index=ignore_index,
                               weight=ce_weights)


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
        self.smooth = smooth
        super(DiceLoss, self).__init__()

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1 - soft_dice_coeff(inp, target, self.smooth)


def get_dice_loss(smooth: float = 1.0):
    return DiceLoss(smooth)