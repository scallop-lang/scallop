import scallopy

from .config import setup_arg_parser, configure

from .bg_blur import bg_blur
from .color_pop import color_pop
from .crop_image import crop_image
from .load_image import load_image
from .save_image import save_image
from .tag_image import tag_image


def load_into_context(ctx: scallopy.Context):
  ctx.register_foreign_function(bg_blur)
  ctx.register_foreign_function(color_pop)
  ctx.register_foreign_function(crop_image)
  ctx.register_foreign_function(load_image)
  ctx.register_foreign_function(save_image)
  ctx.register_foreign_function(tag_image)
