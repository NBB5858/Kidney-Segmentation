import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss_batch_avg(nn.Module):

    # Improves training stability to numerator and denominator over entire batch

    def __init__(self):
        super(DiceLoss_batch_avg, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.sigmoid()

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceLoss_img_avg(nn.Module):

    def __init__(self):
        super(DiceLoss_img_avg, self).__init__()

    def forward(self, output_logits, target):
        assert output_logits.size() == target.size(), "prediction and target shapes must match"

        output = nn.Sigmoid()(output_logits)

        numerator = torch.sum(output * target, dim=(2, 3)) + 1

        denominator = torch.sum(output, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) + 1

        dice_coeff = 2 * numerator / denominator

        print(dice_coeff)

        return 1 - dice_coeff.mean()


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, balance_param=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.balance_param = balance_param

    def forward(self, output_logits, target):

        assert len(output_logits.shape) == len(target.shape)
        assert output_logits.size(0) == target.size(0)
        assert output_logits.size(1) == target.size(1)

        logpt = - F.binary_cross_entropy_with_logits(output_logits, target)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss


class BalancedFocalLoss(nn.Module):
    def __init__(self, gamma=2, pos_class_weight=0.25):
        super(BalancedFocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_class_weight = pos_class_weight

    def forward(self, output_logits, target):
        assert len(output_logits.shape) == len(target.shape)
        assert output_logits.size(0) == target.size(0)
        assert output_logits.size(1) == target.size(1)

        logpt = - F.binary_cross_entropy_with_logits(output_logits, target, reduction='none')
        pt = torch.exp(logpt)

        alphat = torch.ones_like(target) * (1 - self.pos_class_weight)
        alphat[target == 1] = self.pos_class_weight

        balanced_focal_loss = (-alphat * (1 - pt) ** self.gamma * logpt).mean()

        return balanced_focal_loss
