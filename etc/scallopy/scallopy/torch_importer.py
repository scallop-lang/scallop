"""
Attempt to import torch. When `torch` is unfound, use placeholder types
"""

try:
  import torch

  # Has PyTorch
  has_pytorch = True
  Context = torch.nn.Module
  Tensor = torch.Tensor
except:
  class _Context:
    pass
  class _Tensor:
    pass
  has_pytorch = False
  Context = _Context
  Tensor = _Tensor
