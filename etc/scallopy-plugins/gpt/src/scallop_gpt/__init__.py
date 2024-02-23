import scallopy

from . import fa_encoder
from . import fa_extract_info
from . import fa_gpt
from . import ff_gpt
from . import fp_gpt

from .config import setup_arg_parser, configure

def load_into_context(ctx: scallopy.ScallopContext):
  ctx.register_foreign_attribute(fa_extract_info.gpt_extract_info)
  ctx.register_foreign_attribute(fa_gpt.gpt)
  ctx.register_foreign_function(ff_gpt.gpt)
  ctx.register_foreign_predicate(fp_gpt.gpt)
  fa_extract_info.reset_memo()

  if scallopy.torch_tensor_enabled():
    ctx.register_foreign_attribute(fa_encoder.gpt_encoder)
