import torch

def select_along_first_dim(tensor, indices):
    """
    Selects slices along the first dimension of the tensor based on provided indices.

    Args:
    - tensor (torch.Tensor): Input tensor.
    - indices (list or torch.Tensor): List or tensor of indices to select.

    Returns:
    - torch.Tensor: Tensor with selected slices.
    """
    # Convert indices to a PyTorch tensor if it's a list
    if isinstance(indices, list):
        indices = torch.tensor(indices)

    # Use advanced indexing to select slices
    return tensor[indices]

# Testing
a = torch.randn(2, 2, 2, 2)  # Assuming you have 3 slices along the first dimension for this example

# Desired order: [1,2,3,1,2,3]
indices = [1,0,0,0]
b = select_along_first_dim(a, indices)
print(b)