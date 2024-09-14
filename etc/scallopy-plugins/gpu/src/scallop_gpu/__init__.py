from typing import Dict, List, Optional
from argparse import ArgumentParser

import scallopy

_DEVICE = "cpu"

class ScallopGPUPlugin(scallopy.Plugin):
  def __init__(self):
    super().__init__()

  def setup_argparse(self, parser: ArgumentParser):
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--gpu", type=int, default=None, help="The ID of GPU to use")

  def configure(self, args: Dict = ..., unknown_args: List = ...):
    self.configure_device(args["cuda"], args["gpu"])

  def configure_device(self, use_cuda: bool = False, gpu: Optional[int] = None):
    global _DEVICE
    if use_cuda:
      if gpu is not None:
        _DEVICE = f"cuda:{gpu}"
      else:
        _DEVICE = "cuda"
    else:
      _DEVICE = "cpu"

def get_device() -> str:
  global _DEVICE
  return _DEVICE
