import numpy as np
import torch
from torch import nn


class Accuracy:
    """Accuracy metric for segmentation tasks.

    Computes pixel-wise accuracy by comparing predicted class labels
    with ground truth labels.
    """

    def __call__(self, input, target, **kwargs):
        """Calculate accuracy between predictions and targets.

        Args:
            input (torch.Tensor): Predicted logits of shape (batch_size, n_classes, H, W)
            target (torch.Tensor): Ground truth labels of shape (batch_size, H, W)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            float: Pixel-wise accuracy as a fraction between 0 and 1

        """
        input = torch.max(input, 1)[1]
        size = 1
        for i in range(len(input.shape)):
            size = size * input.shape[i]
        return torch.sum(input == target).float() / size

def get_size_of_tensor(a_tensor):
    """Calculate the total number of elements in a tensor.

    Args:
        a_tensor (torch.Tensor): Input tensor

    Returns:
        int: Total number of elements in the tensor

    """
    size = 1
    for i in range(len(a_tensor.shape)):
        size = size*a_tensor.shape[i]
    return size

def dice_loss(pred, target, smooth=1.0, if_mean=True):
    """Compute Dice loss for segmentation.

    The Dice loss is based on the Dice coefficient, which measures overlap
    between predicted and target segmentation masks. It's particularly useful
    for segmentation tasks with class imbalance.

    Args:
        pred (torch.Tensor): Predicted probabilities/logits
        target (torch.Tensor): Ground truth binary masks
        smooth (float): Smoothing factor to avoid division by zero
        if_mean (bool): If True, return mean loss; otherwise return per-sample loss

    Returns:
        torch.Tensor: Dice loss value(s)

    """
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    if if_mean:
        # loss = 0.1*loss[:,0] + 0.7*loss[:,1] + 0.2*loss[:,2]
        # loss = 0.04205177**loss[:,0] + 0.73025561*loss[:,1] + 0.22769263*loss[:,2]
        return (1 - loss).mean()
    else:
        return np.squeeze(loss)

class DomainEnrichLoss:
    """Domain enrichment loss for multi-class segmentation.

    This loss function incorporates domain-specific information to improve
    segmentation performance, particularly for distinguishing between
    different tissue types in medical imaging.
    """

    def __init__(self):
        """Initialize the domain enrichment loss.

        Sets up loss components and hyperparameters for domain-aware training.
        """
        self.loss = nn.CrossEntropyLoss()
        self.alpha = torch.from_numpy(np.asarray(1e6)).float()
        self.beta = torch.from_numpy(np.asarray(1e6)).float()
        self.sigma1 = torch.from_numpy(np.asarray(1e6)).float()
        self.sigma2 = torch.from_numpy(np.asarray(1e6)).float()

    def __call__(self, net, ratio):
        """Compute domain enrichment loss.

        Args:
            net: Neural network model
            ratio (torch.Tensor): Class ratio information for domain adaptation

        Returns:
            torch.Tensor: Computed domain enrichment loss

        """
        # bone
        idx_bone = ratio == 1
        idx_dirt = ratio == 0
        rdn1_bone = net.x_rdn1[idx_bone, 0:8, :, :]
        rdn1_dirt = net.x_rdn1[idx_dirt, 0:8, :, :]
        rdn2_bone = net.x_rdn2[idx_bone, 0:8, :, :]
        rdn2_dirt = net.x_rdn2[idx_dirt, 0:8, :, :]

        if rdn1_bone.is_cuda:
            self.alpha = self.alpha.cuda()
            self.beta = self.beta.cuda()
            self.sigma1 = self.sigma1.cuda()
            self.sigma2 = self.sigma2.cuda()

        rdn1_bone_norm = torch.norm(rdn1_bone)/get_size_of_tensor(rdn1_bone)
        rdn1_dirt_norm = torch.norm(rdn1_dirt)/get_size_of_tensor(rdn1_dirt)

        rdn2_bone_norm = torch.norm(rdn2_bone)/get_size_of_tensor(rdn2_bone)
        rdn2_dirt_norm = torch.norm(rdn2_dirt)/get_size_of_tensor(rdn2_dirt)

        LDF_bone = - (self.alpha * rdn1_bone_norm) + (self.beta * rdn1_dirt_norm)
        LDF_dirt = (self.sigma1 * rdn2_bone_norm) - (self.sigma1 * rdn2_dirt_norm)

        return LDF_bone + LDF_dirt

class DiceOverlap:

    def __init__(self, class_num):
        self.len = class_num

    def __call__(self, input, target):

        input = torch.max(input, 1)[1]

        dice = []

        for i in range(self.len):
            sub_target = torch.zeros(target.shape).cuda()
            sub_target[target == i] = 1
            sub_input = torch.zeros(input.shape).cuda()
            sub_input[input == i] = 1

            tp_idx = target == i

            eps = 0.0001
            tp = torch.sum(sub_input[tp_idx] == sub_target[tp_idx])
            fn = torch.sum(sub_input != sub_target)
            tp = tp.float()
            fn = fn.float()
            result = (2*tp + eps) / (2*tp + fn + eps)
            dice.append(result.cpu().data.numpy())

        return np.asarray(dice)
