from argparse import ArgumentParser
from typing import Dict, List
import sys

import scallopy
from scallop_gpu import get_device

from .clip import get_clip

class ScallopCLIPPlugin(scallopy.Plugin):
  def __init__(
      self,
      default_clip_model_checkpoint: str = "ViT-B/32",
  ):
    super().__init__()

    self._default_clip_model_checkpoint = default_clip_model_checkpoint
    self._clip_model_checkpoint = self._default_clip_model_checkpoint
    self._clip_model = None
    self._clip_preprocess = None

  def setup_argparse(self, parser: ArgumentParser):
    parser.add_argument("--clip-model-checkpoint", type=str, default=self._default_clip_model_checkpoint)

  def configure(self, args: Dict = {}, unknown_args: List = []):
    self._clip_model_checkpoint = args["clip_model_checkpoint"]

  def get_clip_model(self, debug=False):
    if self._clip_model is None:
      try:
        if debug:
          print(f"[scallop-clip] Loading CLIP model `{self._clip_model_checkpoint}`...")
        import clip
        model, preprocess = clip.load(self._clip_model_checkpoint, device=get_device())
        self._clip_model = model
        self._clip_preprocess = preprocess
        if debug:
          print(f"[scallop-clip] Done!")
      except Exception as ex:
        if debug:
          print(ex, file=sys.stderr)
        return None

    if debug:
      print("[scallop-clip] Using loaded CLIP model")

    return (self._clip_model, self._clip_preprocess)

  def load_into_ctx(self, ctx: scallopy.ScallopContext):
    ctx.register_foreign_attribute(get_clip(self))
