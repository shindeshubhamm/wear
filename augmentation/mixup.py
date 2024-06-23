from typing import Tuple

import torch


def apply_mixup(data: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply MixUp augmentation to HAR data.
    
    Parameters:
    data (torch.Tensor): Input data of shape (num_samples, window_size, num_features)
    labels (torch.Tensor): Labels corresponding to the data of shape (num_samples,)
    alpha (float): Parameter controlling the strength of interpolation.
    
    Returns:
    torch.Tensor, torch.Tensor: Augmented data and corresponding labels.
    """
    
    # for mixup, alpha value between 0.1 and 0.4 is ideal.
    # this makes data more closer to original data.
    
    batch_size = data.size(0)
    
    # Generate random indices for mixing
    indices = torch.randperm(batch_size)
    
    # Generate mixing coefficients
    lam = torch.distributions.beta.Beta(alpha, alpha).sample((batch_size,))
    lam = lam.to(data.device)
    
    # Reshape lam to match data dimensions
    lam_reshaped = lam.view(-1, 1, 1)

    # Perform mixup on data
    mixed_data = lam_reshaped * data + (1 - lam_reshaped) * data[indices]
    
    # Perform mixup on labels (keeping them as integers)
    mixed_labels = labels.clone()
    mixed_labels[lam < 0.5] = labels[indices][lam < 0.5]
    
    return mixed_data, mixed_labels


def apply_s3(data: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply MixUp augmentation to arm modality only.
    
    Parameters:
    data (torch.Tensor): Input data of shape (num_samples, window_size, num_features)
    labels (torch.Tensor): Labels corresponding to the data of shape (num_samples,)
    alpha (float): Parameter controlling the strength of interpolation.
    
    Returns:
    torch.Tensor, torch.Tensor: Augmented data and corresponding labels.
    """
    
    batch_size = data.size(0)
    
    # Generate random indices for mixing
    indices = torch.randperm(batch_size)
    
    # Generate mixing coefficients
    lam = torch.distributions.beta.Beta(alpha, alpha).sample((batch_size,))
    lam = lam.to(data.device)
    
    # Reshape lam to match data dimensions
    lam_reshaped = lam.view(-1, 1, 1)

    # Perform mixup on data
    mixed_data = data.clone()
    mixed_data[:, :, :3] = lam_reshaped * data[:, :, :3] + (1 - lam_reshaped) * data[indices, :, :3] # right arm
    mixed_data[:, :, -3:] = lam_reshaped * data[:, :, -3:] + (1 - lam_reshaped) * data[indices, :, -3:] # left arm
    
    # Perform mixup on labels (keeping them as integers)
    mixed_labels = labels.clone()
    mixed_labels[lam < 0.5] = labels[indices][lam < 0.5]
    
    return mixed_data, mixed_labels


def apply_s4(data: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply MixUp augmentation to leg modality only.
    
    Parameters:
    data (torch.Tensor): Input data of shape (num_samples, window_size, num_features)
    labels (torch.Tensor): Labels corresponding to the data of shape (num_samples,)
    alpha (float): Parameter controlling the strength of interpolation.
    
    Returns:
    torch.Tensor, torch.Tensor: Augmented data and corresponding labels.
    """
    
    
    batch_size = data.size(0)

    # Generate random indices for mixing
    indices = torch.randperm(batch_size)

    # Generate mixing coefficients
    lam = torch.distributions.beta.Beta(alpha, alpha).sample((batch_size,))
    lam = lam.to(data.device)

    # Reshape lam to match data dimensions
    lam_reshaped = lam.view(-1, 1, 1)

    # Perform mixup on data
    mixed_data = data.clone()
    mixed_data[:, :, 3:9] = lam_reshaped * data[:, :, 3:9] + (1 - lam_reshaped) * data[indices, :, 3:9] # right and left leg

    # Perform mixup on labels (keeping them as integers)
    mixed_labels = labels.clone()
    mixed_labels[lam < 0.5] = labels[indices][lam < 0.5]

    return mixed_data, mixed_labels
    