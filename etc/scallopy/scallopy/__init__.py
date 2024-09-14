from .attribute import foreign_attribute, ForeignAttributeProcessor
from .context import ScallopContext
from .provenance import ScallopProvenance
from .function import GenericTypeParameter, foreign_function, ForeignFunction
from .predicate import Facts, foreign_predicate, ForeignPredicate
from .value_types import *
from .tag_types import *
from .input_mapping import InputMapping
from .plugin import PluginRegistry, Plugin
from .syntax import *
from .scallopy import torch_tensor_enabled

from . import input_output as io

# Provide a few aliases
Context = ScallopContext
Provenance = ScallopProvenance
Generic = GenericTypeParameter
ff = foreign_function
fp = foreign_predicate
Map = InputMapping

# Loading
def __getattr__(name: str):
  forward_alias = ["ScallopForwardFunction", "ForwardFunction", "Module"]
  if name in forward_alias:
    from .forward import ScallopForwardFunction
    return ScallopForwardFunction
  elif name in globals():
    return globals()[name]
  else:
    raise AttributeError(f"Attribute `{name}` not found inside `scallopy`")
