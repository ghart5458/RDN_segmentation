import albumentations as albu
import numpy as np
import torch

# Image dimension constants
IMAGE_2D_DIMS = 2
IMAGE_3D_DIMS = 3

def create_one_hot(mask, num_classes=3):
    """Create one-hot encoded masks for multi-class segmentation.

    Converts integer class labels to one-hot encoded format for use in
    neural network training. Each class gets its own channel in the output.

    Args:
        mask (torch.Tensor): Integer mask tensor of shape (batch_size, H, W)
        num_classes (int): Number of segmentation classes (default: 3)

    Returns:
        torch.Tensor: One-hot encoded mask of shape (batch_size, num_classes, H, W)

    """
    one_hot_mask = torch.zeros([mask.shape[0],
                                num_classes,
                                mask.shape[1],
                                mask.shape[2]],
                               dtype=torch.float32)
    if mask.is_cuda:
        one_hot_mask = one_hot_mask.cuda()
    one_hot_mask = one_hot_mask.scatter(1, mask.long().data.unsqueeze(1), 1.0)

    return one_hot_mask

def adjustMask(mask, class_num):
    """Adjust mask values to match class indices for segmentation.

    Maps continuous grayscale values to discrete class indices by dividing
    the 0-255 range into equal intervals for each class.

    Args:
        mask (np.ndarray): Input mask with grayscale values
        class_num (int): Number of segmentation classes

    Returns:
        np.ndarray: Adjusted mask with class indices (0 to class_num-1)

    """
    interval = int(256.0 / class_num)

    # Color_Dict must be a numpy type
    # mask.shape must be a H x W x C
    # do not have channel dimensions
    if len(mask.shape) == IMAGE_2D_DIMS:
        new_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.long)
        for i in range(class_num):
            if i <= class_num - 2:
                new_mask[(mask >= i*interval) & (mask < (i+1) * interval)] = i
            else:
                new_mask[i*interval <= mask] = i
        return new_mask

class AdjustMask:
    """Transform class for adjusting mask values in data processing pipelines.

    Callable class that wraps adjustMask function for use in data
    transformation pipelines during training and inference.
    """

    def __init__(self, class_num=3):
        """Initialize mask adjustment transform.

        Args:
            class_num (int): Number of segmentation classes (default: 3)

        """
        self.class_num = class_num

    def __call__(self, sample):
        """Apply mask adjustment to a data sample.

        Args:
            sample (dict): Data sample containing 'mask' key

        Returns:
            dict: Sample with adjusted mask values

        """
        sample['mask'] = adjustMask(sample['mask'], self.class_num)
        return sample

class ToTensor:
    """Convert numpy arrays to PyTorch tensors with proper dimension handling.

    Transforms numpy images and masks to PyTorch tensors, handling different
    input dimensions and multi-image scenarios for training.
    """

    def __init__(self, if_multi_img=False):
        """Initialize tensor conversion transform.

        Args:
            if_multi_img (bool): Whether processing multiple images simultaneously

        """
        self.if_multi_img = if_multi_img

    def __call__(self, sample):
        """Convert sample arrays to PyTorch tensors.

        Args:
            sample (dict): Data sample containing 'image' and 'mask' keys

        Returns:
            dict: Sample with tensors converted for PyTorch training

        """
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W

        if not self.if_multi_img:
            if len(image.shape) == IMAGE_2D_DIMS:
                image = np.expand_dims(image, axis=2)
            image = image.transpose((2, 0, 1))
        else:
            if len(image.shape) == IMAGE_3D_DIMS:
                image = np.expand_dims(image, axis=3)

            image = image.transpose((0, 3, 1, 2))

        sample['image'] =  torch.from_numpy(image)
        sample['mask'] = torch.from_numpy(mask)

        if 'weights' in sample:
            sample['weights'] = torch.from_numpy(sample['weights'])
        if 'ratio' in sample:
            sample['ratio'] = torch.from_numpy(sample['ratio'])
        return sample

class Normalize:
    """Normalize image intensities to a target range.

    Transforms image pixel values from input range to target range,
    commonly used to normalize images to [0, 1] or [-1, 1] ranges.
    """

    def __init__(self, max=255.0, min=0.0, tg_max=1.0, tg_min=0.0):
        """Initialize normalization transform.

        Args:
            max (float): Maximum input intensity value
            min (float): Minimum input intensity value
            tg_max (float): Target maximum intensity value
            tg_min (float): Target minimum intensity value

        """
        self.max = max
        self.min = min
        self.tg_max = tg_max
        self.tg_min = tg_min

    def __call__(self, sample):
        """Apply normalization to image in sample.

        Args:
            sample (dict): Data sample containing 'image' key

        Returns:
            dict: Sample with normalized image

        """
        image = sample['image'].astype('float32')
        image = self.tg_min + ((image - self.min)*(self.tg_max - self.tg_min)) / (self.max - self.min)
        sample['image'] = image
        return sample

class Augmentation:
    """Data augmentation transform for medical image segmentation.

    Applies various augmentation techniques including geometric transformations,
    intensity changes, and elastic deformations to improve model robustness.
    """

    def __init__(self, output_size=256):
        """Initialize augmentation pipeline.

        Args:
            output_size (int): Target image size after augmentation

        """
        self.aug = albu.Compose([
            albu.HorizontalFlip(),
            albu.OneOf([
            albu.RandomContrast(),
            albu.RandomGamma(),
            albu.RandomBrightness(),
            ], p=0.5),
            albu.OneOf([
            albu.ElasticTransform(alpha=60, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            albu.GridDistortion(),
            albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.5),
            albu.ShiftScaleRotate(rotate_limit=180),
            albu.Resize(output_size, output_size, always_apply=True),
        ])
    def __call__(self, sample):
        """Apply augmentation to image and mask simultaneously.

        Args:
            sample (dict): Data sample containing 'image' and 'mask' keys

        Returns:
            dict: Augmented sample with transformed image and mask

        """
        augmented = self.aug(image=sample['image'], mask=sample['mask'])
        sample['image'] = augmented['image']
        sample['mask'] = augmented['mask']
        return sample