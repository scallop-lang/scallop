import os
from argparse import ArgumentParser

from scallop_gpu import get_device
from segment_anything import sam_model_registry


_MODEL_TYPE = "default"

_CHECKPOINT = None

_SAM = None

def setup_arg_parser(parser: ArgumentParser):
  parser.add_argument("--sam-checkpoint", type=str, default=None)


def configure(args):
  global _CHECKPOINT
  try:
    if args["sam_checkpoint"] is not None:
      _CHECKPOINT = args["sam_checkpoint"]
    else:
      _CHECKPOINT = os.getenv("SAM_CHECKPOINT")
    if _CHECKPOINT is None:
      return
  except:
    pass


def get_sam_model():
  global _MODEL_TYPE
  global _CHECKPOINT
  global _SAM

  if _SAM is None:
    _SAM = sam_model_registry[_MODEL_TYPE](checkpoint=_CHECKPOINT).to(device=get_device())
    return _SAM
  else:
    return _SAM
