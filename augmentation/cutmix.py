from typing import Tuple

import torch

def apply_cutmix(data: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply CutMix augmentation to HAR data.
    
    Parameters:
    data (torch.Tensor): Input data of shape (num_samples, window_size, num_features)
    labels (torch.Tensor): Labels corresponding to the data of shape (num_samples,)
    alpha (float): Parameter for the beta distribution. Controls the area to cut and mix.
    
    Returns:
    torch.Tensor, torch.Tensor: Augmented data and corresponding labels.
    """
    # Check if data and labels have the correct shape
    if len(data.shape) != 3 or len(labels.shape) != 1:
        raise ValueError("Data should be of shape (num_samples, window_size, num_features) and labels of shape (num_samples,)")

    indices = torch.randperm(data.shape[0])
    
    cut_data = data.clone()
    cut_labels = labels.clone()
    
    for i in range(data.shape[0]):
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
        
        # Choose the length of the segment to be replaced
        cut_length = int(lam * data.shape[1])
        
        # Random start position for the segment
        start = torch.randint(0, data.shape[1] - cut_length + 1, (1,)).item()
        
        # Choose a random sample to mix with
        j = indices[i]
        
        cut_data[i, start:start + cut_length, :] = data[j, start:start + cut_length, :]
        
        # Randomly choose one of the labels
        if torch.rand(1).item() < lam:
            cut_labels[i] = labels[j]
    
    return cut_data, cut_labels


def apply_s1(data: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply CutMix augmentation to arm modality only.
    
    Parameters:
    data (torch.Tensor): Input data of shape (num_samples, window_size, num_features)
    labels (torch.Tensor): Labels corresponding to the data of shape (num_samples,)
    alpha (float): Parameter for the beta distribution. Controls the area to cut and mix.
    
    Returns:
    torch.Tensor, torch.Tensor: Augmented data and corresponding labels.
    """
    # Check if data and labels have the correct shape
    if len(data.shape) != 3 or len(labels.shape) != 1:
        raise ValueError("Data should be of shape (num_samples, window_size, num_features) and labels of shape (num_samples,)")

    indices = torch.randperm(data.shape[0])
    
    cut_data = data.clone()
    cut_labels = labels.clone()

    for i in range(data.shape[0]):
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
        
        # Choose the length of the segment to be replaced
        cut_length = int(lam * data.shape[1])
        
        # Random start position for the segment
        start = torch.randint(0, data.shape[1] - cut_length + 1, (1,)).item()
        
        # Choose a random sample to mix with
        j = indices[i]
        
        cut_data[i, start:start + cut_length, :3] = data[j, start:start + cut_length, :3]
        cut_data[i, start:start + cut_length, -3:] = data[j, start:start + cut_length, -3:]
        
        # Randomly choose one of the labels
        if torch.rand(1).item() < lam:
            cut_labels[i] = labels[j]
    
    return cut_data, cut_labels


def apply_s2(data: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply CutMix augmentation to leg modality only.
    
    Parameters:
    data (torch.Tensor): Input data of shape (num_samples, window_size, num_features)
    labels (torch.Tensor): Labels corresponding to the data of shape (num_samples,)
    alpha (float): Parameter for the beta distribution. Controls the area to cut and mix.
    
    Returns:
    torch.Tensor, torch.Tensor: Augmented data and corresponding labels.
    """
    # Check if data and labels have the correct shape
    if len(data.shape) != 3 or len(labels.shape) != 1:
        raise ValueError("Data should be of shape (num_samples, window_size, num_features) and labels of shape (num_samples,)")

    indices = torch.randperm(data.shape[0])
    
    cut_data = data.clone()
    cut_labels = labels.clone()

    for i in range(data.shape[0]):
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
        
        # Choose the length of the segment to be replaced
        cut_length = int(lam * data.shape[1])
        
        # Random start position for the segment
        start = torch.randint(0, data.shape[1] - cut_length + 1, (1,)).item()
        
        # Choose a random sample to mix with
        j = indices[i]
        
        cut_data[i, start:start + cut_length, 3:9] = data[j, start:start + cut_length, 3:9]
        
        # Randomly choose one of the labels
        if torch.rand(1).item() < lam:
            cut_labels[i] = labels[j]
    
    return cut_data, cut_labels