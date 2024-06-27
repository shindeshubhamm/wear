from typing import Tuple

import torch


def apply_jittering(x: torch.Tensor, labels: torch.Tensor, alpha: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Jittering augmentation to HAR data.
    
    Parameters:
    data (torch.Tensor): Input data of shape (num_samples, window_size, num_features)
    labels (torch.Tensor): Labels corresponding to the data of shape (num_samples,)
    alpha (float): Parameter for the beta distribution. Controls the area to cut and mix.
    
    Returns:
    torch.Tensor, torch.Tensor: Augmented data and corresponding labels.
    """
    
    
    # Ensure the input tensor is on CPU
    x_cpu = x.cpu() if x.is_cuda else x

    # Generate jitter noise
    jitter_noise = torch.randn_like(x_cpu) * sigma

    # Add jitter noise to the input tensor
    jittered_x = x_cpu + jitter_noise

    # If the input tensor was originally on CUDA, move the result back to CUDA
    if x.is_cuda:
        jittered_x = jittered_x.to(x.device)

    return jittered_x,labels

def apply_s1_jitter(x: torch.Tensor, labels: torch.Tensor, sigma: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Jittering augmentation to HAR data.

    Parameters:
    x (torch.Tensor): Input data of shape (num_samples, window_size, num_features)
    y (torch.Tensor): Labels corresponding to the data of shape (num_samples,)
    sigma (float): Standard deviation of the jitter noise.

    Returns:
    torch.Tensor, torch.Tensor: Augmented data and corresponding labels.
    """

    # Ensure the input tensor is on CPU
    x_cpu = x.cpu() if x.is_cuda else x

    # Apply jitter noise only to the first 3 and last 3 columns
    num_features = x_cpu.size(-1)
    jitter_noise = torch.randn_like(x_cpu) * sigma
    jitter_noise[:, :, :3] = 0.0  # Zero out jitter noise for columns 0, 1, 2
    jitter_noise[:, :, -3:] = 0.0  # Zero out jitter noise for last 3 columns
    jittered_x = x_cpu + jitter_noise

    # If the input tensor was originally on CUDA, move the result back to CUDA
    if x.is_cuda:
        jittered_x = jittered_x.to(x.device)

    return jittered_x, labels

def apply_s2_jitter(x: torch.Tensor, labels: torch.Tensor, sigma: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Jittering augmentation to HAR data.

    Parameters:
    x (torch.Tensor): Input data of shape (num_samples, window_size, num_features)
    y (torch.Tensor): Labels corresponding to the data of shape (num_samples,)
    sigma (float): Standard deviation of the jitter noise.

    Returns:
    torch.Tensor, torch.Tensor: Augmented data and corresponding labels.
    """

    # Ensure the input tensor is on CPU
    x_cpu = x.cpu() if x.is_cuda else x

    # Generate jitter noise
    jitter_noise = torch.zeros_like(x_cpu)  # Initialize with zeros
    jitter_noise[:, :, 6:9] = torch.randn_like(x_cpu[:, :, 6:9]) * sigma  # Add noise to columns 6 to 9

    # Add jitter noise to the input tensor
    jittered_x = x_cpu + jitter_noise

    # If the input tensor was originally on CUDA, move the result back to CUDA
    if x.is_cuda:
        jittered_x = jittered_x.to(x.device)

    return jittered_x, labels
 
