import numpy as np
import torch
from typing import Tuple

# import augmentation methods
from .cutmix import apply_cutmix, apply_s1, apply_s2
from .mixup import apply_mixup, apply_s3, apply_s4
from .jittering import apply_jittering

def apply_augmentation(augmentation_choice: str, data: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply augmentation technique based on the given choice.

    Parameters:
    - augmentation_choice (str): The augmentation technique to apply.
    - data (torch.Tensor): The input data array to which augmentation will be applied.
    - labels (torch.Tensor): The corresponding labels array.

    Returns:
    torch.Tensor, torch.Tensor: Augmented data and labels. If the augmentation_choice is not recognized or specified, returns the original data and labels unchanged.

    Example:
    >>> data_augmented, labels_augmented = apply_augmentation('cutmix', data, labels)
    """
    match augmentation_choice:
        case 'cutmix':
            return apply_cutmix(data, labels)
        
        case 's1': # apply cutmix to arm modality only
            return apply_s1(data, labels)
        
        case 's2': # apply cutmix to leg modality only
            return apply_s2(data, labels)
        
        case 'mixup':
            return apply_mixup(data, labels)
        
        case 's3': # apply mixup to arm modality only
            return apply_s3(data, labels)

        case 's4': # apply mixup to leg modality only
            return apply_s4(data, labels)
        
        case 'jittering':
            return apply_jittering(data, labels)

        case 'js1':  # apply cutmix to leg modality only
            return apply_s1_jitter(data, labels)

        case 'js2':  # apply cutmix to leg modality only
            return apply_s2_jitter(data, labels)

        case _:
            return data, labels
