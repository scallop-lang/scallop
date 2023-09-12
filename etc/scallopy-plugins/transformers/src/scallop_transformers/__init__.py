import scallopy

from .vilt import vilt
from .owl_vit import owl_vit
from .roberta_encoder import roberta_encoder

def load_into_context(ctx: scallopy.Context):
  ctx.register_foreign_attribute(vilt)
  ctx.register_foreign_attribute(owl_vit)

  ctx.register_foreign_attribute(roberta_encoder)
