import scallopy

from .config import setup_arg_parser, configure
from .clip import clip

def load_into_context(ctx: scallopy.Context):
  ctx.register_foreign_attribute(clip)
