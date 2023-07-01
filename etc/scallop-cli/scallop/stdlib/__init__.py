from scallopy import ScallopContext

from . import ff
from . import fp
from . import attr

def load_stdlib(scallop_ctx: ScallopContext):
  # Register foreign functions
  scallop_ctx.register_foreign_function(ff.gpt_complete)

  # Register foreign predicates
  scallop_ctx.register_foreign_predicate(fp.gpt_complete)

  # Register foreign attributes
  scallop_ctx.register_foreign_attribute(attr.gpt_complete)
  scallop_ctx.register_foreign_attribute(attr.gpt_chat_complete)
