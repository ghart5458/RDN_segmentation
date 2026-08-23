import re

import h5py
import numpy as np
import pandas as pd
import utils.dataprocess as dp
from torch.utils.data import Dataset
from torchvision import transforms

# Magic numbers used for class comparison
NUM_CLASSES_3 = 3

def alpha_to_int(text):
    """Convert text to integer if it represents a digit, otherwise return as-is.

    Used for natural sorting of strings that may contain numbers.

    Args:
        text (str): Input text that may represent an integer

    Returns:
        int or str: Integer value if text is numeric, otherwise original text

    """
    clean_text = int(text) if text.isdigit() else text
    return clean_text

def alpha_to_float(text):
    """Convert text to float if possible, otherwise return as-is.

    Used for natural sorting of strings that may contain floating point numbers.

    Args:
        text (str): Input text that may represent a float

    Returns:
        float or str: Float value if text is numeric, otherwise original text

    """
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    """Generate keys for natural sorting of strings with embedded numbers.

    Enables human-friendly sorting (e.g., ['file1', 'file2', 'file10'] instead
    of ['file1', 'file10', 'file2']).

    Args:
        text (str): Input string to generate sorting key for

    Returns:
        list: List of mixed integers and strings for natural sorting

    Example:
        >>> files = ['file1.txt', 'file10.txt', 'file2.txt']
        >>> files.sort(key=natural_keys)
        >>> print(files)  # ['file1.txt', 'file2.txt', 'file10.txt']

    Reference:
        http://nedbatchelder.com/blog/200712/human_sorting.html

    """
    return [alpha_to_int(c) for c in re.split(r'(\d+)', text)]

def natural_keys_float(text):
    """Generate keys for natural sorting of strings with embedded floating point numbers.

    Similar to natural_keys but handles floating point numbers in addition to integers.

    Args:
        text (str): Input string to generate sorting key for

    Returns:
        list: List of mixed floats and strings for natural sorting

    Reference:
        http://nedbatchelder.com/blog/200712/human_sorting.html
        Float regex from https://stackoverflow.com/a/12643073/190597

    """
    return [alpha_to_float(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]

def load_patches(patches):
    """Load patch information from CSV file or return existing patch data.

    Args:
        patches (str or list): Either a path to CSV file containing patches,
                              or existing patch data as list

    Returns:
        list: Patch information as list of lists containing patch coordinates

    """
    if isinstance(patches, str):
        return np.array(pd.read_csv(patches, header=0)).tolist()
    else:
        return patches

class HDF52D(Dataset):
    """PyTorch Dataset class for loading 2D image patches from HDF5 files.

    This dataset is designed for segmentation tasks where images and corresponding
    masks are stored in HDF5 format. It supports separate training and validation
    patch sets with different transforms.

    Attributes:
        data_path (str): Path to HDF5 file containing images and masks
        patches (dict): Dictionary containing 'train' and 'val' patch lists
        transforms (dict): Dictionary containing 'train' and 'val' transforms
        train_idx (list): Optional training indices for tracking
        mode (str): Current mode - 'train' or 'val'

    """

    def __init__(self, data_path, train_patches, val_patches, train_transform=None, val_transform=None, train_idx=None):
        """Initialize HDF5 2D dataset.

        Args:
            data_path (str): Path to HDF5 file containing image data
            train_patches (str or list): Training patch coordinates (CSV path or list)
            val_patches (str or list): Validation patch coordinates (CSV path or list)
            train_transform (callable, optional): Transform to apply to training samples
            val_transform (callable, optional): Transform to apply to validation samples
            train_idx (str or list, optional): Training indices for tracking (CSV path or list)

        """
        self.data_path = data_path

        self.patches = {'train': load_patches(train_patches),
                        'val': load_patches(val_patches)}

        self.transforms = {'train': train_transform,
                           'val': val_transform}

        self.train_idx = load_patches(train_idx)

        self.mode = 'train'

    def __getitem__(self, idx):
        """Get a single data sample by index.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            dict: Sample dictionary containing:
                - 'image': Image patch as numpy array
                - 'mask': Corresponding mask as numpy array
                - 'index': Training index (if available and in train mode)

        """
        [name, top, left, h, w] = self.patches[self.mode][idx]

        with h5py.File(self.data_path,'r') as f:
            image = f[name]['data'][top:top + h, left:left + w ]
            mask = f[name]['label'][top:top + h, left:left + w]
            sample = {'image': image, 'mask': mask}

        if self.transforms[self.mode] is not None:
            sample = self.transforms[self.mode](sample)
        if self.train_idx is not None and self.mode == 'train':
            sample['index'] = self.train_idx[idx]
        return sample

    def train(self):
        """Switch dataset to training mode."""
        self.mode = 'train'

    def val(self):
        """Switch dataset to validation mode."""
        self.mode = 'val'

    def __len__(self):
        """Get the length of the current dataset split.

        Returns:
            int: Number of samples in current mode (train or val)

        """
        return len(self.patches[self.mode])

if __name__ == '__main__':
    # Example usage and testing
    data_path = '/cvdata/yungchen/rdn_revised/data/dataset.hdf5'
    train_patches = '/cvdata/yungchen/rdn_revised/data/patches.csv'
    val_patches = '/cvdata/yungchen/rdn_revised/data/val.csv'
    ratios = '/cvdata/yungchen/rdn_revised/data/ratios.csv'

    transforms = transforms.Compose([dp.Augmentation(output_size=256),
                                     dp.AdjustMask(class_num=NUM_CLASSES_3),
                                     dp.Normalize(input_max=255, input_min=0)])

    data_set = HDF52D(data_path,train_patches,val_patches,train_transform=transforms, train_idx=ratios)
    sample = data_set[1000]
    mask = sample['mask']

    print(np.sum(mask == 0))
    print(np.sum(mask == 1))
    print(np.sum(mask == 2))
