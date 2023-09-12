import scallopy

from .config import setup_arg_parser, configure
from .sam import segment_anything

def load_into_context(ctx: scallopy.Context):
  ctx.register_foreign_attribute(segment_anything)
