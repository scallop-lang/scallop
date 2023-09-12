from typing import ForwardRef

from . import value_types

none = "none"
natural = "natural"
prob = "prob"
exclusion = "exclusion"
boolean = "bool"
exclusive_prob = "exclusive-prob"
diff_prob = "diff-prob"
exclusive_diff_prob = "exclusive-diff-prob"

ALL_TAG_TYPES = set([
  prob,
  exclusion,
  bool,
  exclusive_prob,
  diff_prob,
  exclusive_diff_prob,
])

def get_tag_type(v) -> str:
  if v is None or v is type(None):
    return none
  elif v is int:
    return natural
  elif v is bool:
    return boolean
  elif v is float:
    return prob
  elif v == value_types.Tensor:
    return diff_prob
  elif v in ALL_TAG_TYPES:
    return v
  elif type(v) == ForwardRef:
    return get_tag_type(v.__forward_arg__)
  else:
    raise Exception(f"Unknown tag type {v}")
