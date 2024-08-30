import scallopy

from .config import setup_arg_parser, configure
from .plip import plip

def load_into_context(ctx: scallopy.Context):
  ctx.register_foreign_attribute(plip)
