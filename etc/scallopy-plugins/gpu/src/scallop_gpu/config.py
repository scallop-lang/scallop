from typing import Optional
from argparse import ArgumentParser


_DEVICE = "cpu"


def setup_arg_parser(parser: ArgumentParser):
  parser.add_argument("--cuda", action="store_true", help="Use CUDA")
  parser.add_argument("--gpu", type=int, default=None, help="The ID of GPU to use")


def configure(args):
  configure_device(args.cuda, args.gpu)


def configure_device(use_cuda: bool = False, gpu: Optional[int] = None):
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
