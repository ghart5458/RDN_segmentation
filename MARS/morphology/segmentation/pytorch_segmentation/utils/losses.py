import numpy as np
import torch
from torch import nn


class Accuracy:
    """Accuracy metric for segmentation tasks.

    Computes pixel-wise accuracy by comparing predicted class labels
    with ground truth labels.
    """

    def __call__(self, predictions, target, **kwargs):
        """Calculate accuracy between predictions and targets.

        Args:
            predictions (torch.Tensor): Predicted logits of shape (batch_size, n_classes, H, W)
            target (torch.Tensor): Ground truth labels of shape (batch_size, H, W)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            float: Pixel-wise accuracy as a fraction between 0 and 1

        """
        predictions = torch.max(predictions, 1)[1]
        size = 1
        for i in range(len(predictions.shape)):
            size = size * predictions.shape[i]
        return torch.sum(predictions == target).float() / size

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
        self.alpha = torch.from_numpy(np.asarray(1e0)).float()
        self.beta = torch.from_numpy(np.asarray(1e0)).float()
        self.gamma = torch.from_numpy(np.asarray(1e0)).float()
        self.sigma = torch.from_numpy(np.asarray(1e0)).float()
        self.zeta = torch.from_numpy(np.asarray(1e0)).float()

        # The 1e-4 factor that used to be applied by the caller (loss2 + 1e-4*loss1)
        # is folded in here, squared, matching the IA-SeReOs formulation. The caller
        # must therefore add loss1 unweighted: loss = loss2 + loss1.
        self.lambda1 = 0.0001 * 0.0001
        self.lambda2 = 0.0001 * 0.0001

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
            self.gamma = self.gamma.cuda()
            self.sigma = self.sigma.cuda()
            self.zeta = self.zeta.cuda()

        # Squared L2 norms, no longer divided by tensor size (IA-SeReOs change:
        # the size division was not in the published formulation).
        rdn1_bone_norm2 = torch.norm(rdn1_bone, p=2)
        rdn1_dirt_norm2 = torch.norm(rdn1_dirt, p=2)

        rdn2_bone_norm2 = torch.norm(rdn2_bone, p=2)
        rdn2_dirt_norm2 = torch.norm(rdn2_dirt, p=2)

        # L1 term: this is the sparsity regularizer from "Multi-Class Micro-CT
        # Image Segmentation Using Sparse Regularized Deep Networks".
        rdn2_dirt_norm1 = torch.norm(rdn2_dirt, p=1)

        LDF_bone = -(self.alpha * rdn1_bone_norm2**2) + (self.beta * rdn1_dirt_norm2**2)
        LDF_dirt = (
            (self.gamma * rdn2_bone_norm2**2)
            - (self.sigma * rdn2_dirt_norm2**2)
            + (self.zeta * rdn2_dirt_norm1)
        )

        return torch.sigmoid(self.lambda1 * LDF_bone + self.lambda2 * LDF_dirt)

class DiceOverlap:

    def __init__(self, class_num):
        self.len = class_num

    def __call__(self, predictions, target):

        predictions = torch.max(predictions, 1)[1]

        dice = []

        for i in range(self.len):
            sub_target = torch.zeros(target.shape).cuda()
            sub_target[target == i] = 1
            sub_predictions = torch.zeros(predictions.shape).cuda()
            sub_predictions[predictions == i] = 1

            tp_idx = target == i

            eps = 0.0001
            tp = torch.sum(sub_predictions[tp_idx] == sub_target[tp_idx])
            fn = torch.sum(sub_predictions != sub_target)
            tp = tp.float()
            fn = fn.float()
            result = (2*tp + eps) / (2*tp + fn + eps)
            dice.append(result.cpu().data.numpy())

        return np.asarray(dice)
