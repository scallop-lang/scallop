from argparse import ArgumentParser
from typing import Dict, List
import scallopy

from .bg_blur import bg_blur
from .color_pop import color_pop
from .crop_image import crop_image
from .load_image import load_image, load_image_url
from .save_image import _save_image
from .tag_image import tag_image
from .upload_image import _upload_imgur

class ScallopOpenCVPlugin(scallopy.Plugin):
  def __init__(
      self,
      default_save_image_path: str = ".tmp/scallop-save-image"
  ):
    super().__init__()

    self._default_save_image_path = default_save_image_path
    self._save_image_path = self._default_save_image_path

  def setup_argparse(self, parser: ArgumentParser):
    parser.add_argument("--save-image-path", type=str, default=self._default_save_image_path)

  def configure(self, args: Dict = {}, unknown_args: List = []):
    self._save_image_path = args["save_image_path"]

  def load_into_ctx(self, ctx: scallopy.ScallopContext):
    self.load_save_image_into_ctx(ctx)
    self.load_upload_imgur_into_ctx(ctx)
    self.load_tag_image_into_ctx(ctx)
    self.load_load_image_into_ctx(ctx)
    self.load_crop_image_into_ctx(ctx)
    self.load_color_pop_into_ctx(ctx)
    self.load_bg_blur_into_ctx(ctx)

  def load_save_image_into_ctx(self, ctx: scallopy.ScallopContext):
    @scallopy.foreign_function
    def save_image(img_tensor: scallopy.Tensor, img_name: str = None) -> str:
      return _save_image(img_tensor, img_name, self._save_image_path)
    ctx.register_foreign_function(save_image)

  def load_upload_imgur_into_ctx(self, ctx: scallopy.ScallopContext):
    @scallopy.foreign_function
    def upload_imgur(img_tensor: scallopy.Tensor) -> str:
      return _upload_imgur(img_tensor, self._save_image_path)
    ctx.register_foreign_function(upload_imgur)

  def load_tag_image_into_ctx(self, ctx: scallopy.ScallopContext):
    ctx.register_foreign_function(tag_image)

  def load_load_image_into_ctx(self, ctx: scallopy.ScallopContext):
    ctx.register_foreign_function(load_image)
    ctx.register_foreign_function(load_image_url)

  def load_crop_image_into_ctx(self, ctx: scallopy.ScallopContext):
    ctx.register_foreign_function(crop_image)

  def load_color_pop_into_ctx(self, ctx: scallopy.ScallopContext):
    ctx.register_foreign_function(color_pop)

  def load_bg_blur_into_ctx(self, ctx: scallopy.ScallopContext):
    ctx.register_foreign_function(bg_blur)
