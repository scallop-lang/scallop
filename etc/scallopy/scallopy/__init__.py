from .context import ScallopContext
from .forward import ScallopForwardFunction
from .provenance import ScallopProvenance
from .function import GenericTypeParameter, foreign_function
from .types import *

# Provide a few aliases
Context = ScallopContext
ForwardFunction = ScallopForwardFunction
Provenance = ScallopProvenance
Generic = GenericTypeParameter
ff = foreign_function
