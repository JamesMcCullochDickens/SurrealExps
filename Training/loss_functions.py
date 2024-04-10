import torch.nn as nn
import torch
import torch.nn.functional as F

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


class sampled_ce_loss(nn.Module):
    def __init__(self, samples_per_im: int = 5000, lambds_=(1/6, 5/6)):
        super().__init__()
        self.samples_per_im = samples_per_im
        self.lambds_ = lambds_

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        bs = pred.shape[0]
        expected_samples = self.samples_per_im*bs

        gt = gt.float()
        gt_flat = gt.reshape(-1)
        non_zero_mask_flat = torch.where(gt_flat != 0,
                                         torch.tensor([1.0], device=gt.device),
                                         torch.tensor([0.0], device=gt.device))
        num_zero = int(torch.sum(1-non_zero_mask_flat).cpu().item())
        num_non_zero = int(torch.sum(non_zero_mask_flat).cpu().item())
        num_samples = min(num_zero, num_non_zero, expected_samples)

        if num_samples:
            zero_indices = torch.multinomial(1 - non_zero_mask_flat, num_samples//2, replacement=False)
            non_zero_indices = torch.multinomial(non_zero_mask_flat, num_samples, replacement=False)

            num_classes = pred.shape[1]
            pred = torch.permute(pred, (1, 0, 2, 3))
            pred = pred.reshape(num_classes, -1)

            gt_flat = gt_flat.long()

            gt_zero = gt_flat[zero_indices]
            gt_zero = torch.unsqueeze(gt_zero, dim=0)
            pred1 = pred[:, zero_indices]
            pred1 = torch.unsqueeze(pred1, dim=0)

            gt_non_zero = gt_flat[non_zero_indices]
            gt_non_zero = torch.unsqueeze(gt_non_zero, dim=0)
            pred2 = pred[:, non_zero_indices]
            pred2 = torch.unsqueeze(pred2, dim=0)

            loss1 = F.cross_entropy(pred1, gt_zero)
            loss2 = F.cross_entropy(pred2, gt_non_zero)
            total_loss = self.lambds_[0]*loss1 + self.lambds_[1]*loss2

        else:
            total_loss = F.cross_entropy(pred, gt.long())

        return total_loss


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