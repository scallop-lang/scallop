from .context import ScallopContext
from .forward import ScallopForwardFunction
from .provenance import ScallopProvenance
from .function import GenericTypeParameter, foreign_function
from .predicate import Generator, foreign_predicate
from .types import *
from .input_mapping import InputMapping

# Provide a few aliases
Context = ScallopContext
ForwardFunction = ScallopForwardFunction
Module = ScallopForwardFunction
Provenance = ScallopProvenance
Generic = GenericTypeParameter
ff = foreign_function
fp = foreign_predicate
Map = InputMapping
