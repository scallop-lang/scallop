"""
Attempt to import torch. When `torch` is unfound, use placeholder types
"""

import sys

class _DummyModule: pass

class _DummyTensor: pass

_Module = _DummyModule

_Tensor = _DummyTensor

_torch = None

_has_pytorch = None

def has_pytorch():
  try_import_pytorch()
  return _has_pytorch


def try_import_pytorch():
  global _torch
  global _Module
  global _Tensor
  global _has_pytorch

  if _has_pytorch is not None:
    return

  try:
    import torch as curr_torch

    _torch = curr_torch
    _Module = curr_torch.nn.Module
    _Tensor = curr_torch.Tensor

    # Successfully loaded
    _has_pytorch = True
  except:
    # Have not loaded
    _has_pytorch = False


def get_torch():
  global _torch
  return _torch


def get_module():
  global _Module
  return _Module


def get_tensor():
  global _Tensor
  return _Tensor


def __getattr__(name: str):
  trigger_load_items = [
    "torch",
    "Module",
  ]

  passive_load_items = [
    "Tensor"
  ]

  items = {
    "torch": get_torch,
    "Module": get_module,
    "Tensor": get_tensor,
  }

  check_validity = {
    "torch": lambda x: x is not None,
    "Module": lambda x: type(x) != _Module,
    "Tensor": lambda x: type(x) != _Tensor,
  }

  if name in trigger_load_items:
    try_import_pytorch()
    item = items[name]()
    if check_validity[name](item):
      return item
    else:
      raise Exception(f"`torch` is not imported successfully; consider installing torch")
  elif name in passive_load_items:
    if "torch" in sys.modules:
      try_import_pytorch()
    return items[name]()
  else:
    raise Exception(f"Unknown item from `torch_importer`: `{name}`")
