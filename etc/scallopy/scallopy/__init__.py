from .context import ScallopContext
from .forward import ScallopForwardFunction
from .provenance import ScallopProvenance
from .function import GenericTypeParameter, foreign_function, ForeignFunction
from .predicate import Generator, foreign_predicate, ForeignPredicate
from .types import *
from .input_mapping import InputMapping
from .scallopy import torch_tensor_enabled
from .attribute import foreign_attribute, ForeignAttributeProcessor

# Provide a few aliases
Context = ScallopContext
ForwardFunction = ScallopForwardFunction
Module = ScallopForwardFunction
Provenance = ScallopProvenance
Generic = GenericTypeParameter
ff = foreign_function
fp = foreign_predicate
Map = InputMapping
