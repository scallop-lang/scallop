from collections.abc import Generator
from typing import (
  TypeVar,
  Union,
  Tuple,
  Callable,
  List,
  Optional,
  Any,
  ForwardRef,
  ClassVar,
  get_origin,
  get_args,
  get_type_hints,
  TypeAlias,
)
from types import NoneType
import inspect

from . import torch_importer
from . import syntax
from . import utils
from . import tag_types


ALIASES = {
  "I8": "i8",
  "I16": "i16",
  "I32": "i32",
  "I64": "i64",
  "I128": "i128",
  "ISize": "isize",
  "U8": "u8",
  "U16": "u16",
  "U32": "u32",
  "U64": "u64",
  "U128": "u128",
  "USize": "usize",
  "F32": "f32",
  "F64": "f64",
  "Char": "char",
  "Bool": "bool",
}


# Predicate Data Type
class Type:
  def __init__(self, value):
    if isinstance(value, ForwardRef):
      self.type = Type.sanitize_type_str(value.__forward_arg__)
    elif isinstance(value, syntax.AstTypeNode):
      self.type = value.name()
    elif isinstance(value, TypeVar):
      self.type = Type.sanitize_type_str(value.__name__)
    elif isinstance(value, str):
      self.type = Type.sanitize_type_str(value)
    elif value == float:
      self.type = "f32"
    elif value == int:
      self.type = "i32"
    elif value == bool:
      self.type = "bool"
    elif value == str:
      self.type = "String"
    elif value == torch_importer.Tensor:
      self.type = "Tensor"
    else:
      raise TypeError(f"Unknown scallop predicate type annotation `{value}`")

  def __repr__(self):
    return self.type

  @staticmethod
  def sanitize_type_str(value):
    if (
      value == "i8"
      or value == "i16"
      or value == "i32"
      or value == "i64"
      or value == "i128"
      or value == "isize"
      or value == "u8"
      or value == "u16"
      or value == "u32"
      or value == "u64"
      or value == "u128"
      or value == "usize"
      or value == "f32"
      or value == "f64"
      or value == "bool"
      or value == "char"
      or value == "String"
      or value == "Symbol"
      or value == "Tensor"
      or value == "DateTime"
      or value == "Duration"
      or value == "Entity"
    ):
      return value
    elif value in ALIASES:
      return ALIASES[value]
    else:
      raise TypeError(f"Unknown scallop predicate type annotation `{value}`")


TagType = TypeVar("TagType")

TupleType = TypeVar("TupleType")

Facts: TypeAlias = ClassVar[
  Generator[
    Union[
      Tuple[TagType, TupleType],
      TupleType,
    ],
    None,
    None,
  ]
]


class ForeignPredicate:
  """
  Scallop foreign predicate
  """

  def __init__(
    self,
    func: Callable,
    name: str,
    type_params: List[Type],
    input_arg_types: List[Type],
    output_arg_types: List[Type],
    tag_type: str,
    suppress_warning: bool = False,
  ):
    self.func = func
    self.name = name
    self.type_params = type_params
    self.input_arg_types = input_arg_types
    self.output_arg_types = output_arg_types
    self.tag_type = tag_type
    self.suppress_warning = suppress_warning

  def __repr__(self):
    r = f"extern pred {self.name}"
    if len(self.type_params) > 0:
      r += "<"
      for i, type_param in enumerate(self.type_params):
        if i > 0:
          r += ", "
        r += f"{type_param}"
      r += ">"
    r += f"[{self.pattern()}]("
    first = True
    for arg in self.input_arg_types:
      if first:
        first = False
      else:
        r += ", "
      r += f"{arg}"
    for arg in self.output_arg_types:
      if first:
        first = False
      else:
        r += ", "
      r += f"{arg}"
    r += ")"
    return r

  def __call__(self, *args):
    if self.does_output_tag():
      return [f for f in self.func(*args)]
    else:
      return [(None, f) for f in self.func(*args)]

  def arity(self):
    return len(self.input_arg_types) + len(self.output_arg_types)

  def num_bounded(self):
    return len(self.input_arg_types)

  def all_argument_types(self):
    return self.input_arg_types + self.output_arg_types

  def pattern(self):
    return "b" * len(self.input_arg_types) + "f" * len(self.output_arg_types)

  def does_output_tag(self):
    return self.tag_type is not tag_types.none


@utils.doublewrap
def foreign_predicate(
  func: Callable,
  name: Optional[str] = None,
  type_params: Optional[List] = None,
  input_arg_types: Optional[List] = None,
  output_arg_types: Optional[List] = None,
  tag_type: Optional[Any] = None,
  suppress_warning: bool = False,
):
  """
  A decorator to create a Scallop foreign predicate, for example

  ``` python
  @scallopy.foreign_function
  def string_chars(s: str) -> scallopy.Facts[Tuple[int, char]]:
    for (i, c) in enumerate(s):
      yield (i, c)
  ```
  """

  # Get the function name
  func_name = func.__name__ if not name else name

  # Get the function signature
  signature = inspect.signature(func)
  type_hints = get_type_hints(func, include_extras=True)

  # Store all the type params
  if type_params is None:
    processed_type_params = []
  else:
    processed_type_params = [Type(type_param) for type_param in type_params]

  # Store all the argument types
  if input_arg_types is None:
    argument_types = []

    # Find argument types
    for arg_name, item in signature.parameters.items():
      optional = item.default != inspect.Parameter.empty
      if item.annotation is inspect.Parameter.empty or arg_name not in type_hints:
        raise TypeError(
          f"Argument `{arg_name}`'s type annotation not provided in the foreign predicate `{func_name}`"
        )
      if item.kind == inspect.Parameter.VAR_POSITIONAL:
        raise TypeError("Cannot have variable arguments in foreign predicate")
      elif not optional:
        ty = Type(type_hints[arg_name])
        argument_types.append(ty)
      else:
        raise TypeError("Cannot have optional argument in foreign predicate")
  else:
    argument_types = [Type(t) for t in input_arg_types]

  # Find return type
  if output_arg_types is None:
    if (
      signature.return_annotation is inspect.Signature.empty
      or "return" not in type_hints
    ):
      raise TypeError("Return type annotation not provided")

    return_tag_type, return_tuple_type = _extract_facts_return_type(
      type_hints["return"]
    )
  else:
    return_tag_type = tag_types.get_tag_type(tag_type)
    return_tuple_type = [Type(t) for t in output_arg_types]

  # Create the foreign predicate
  return ForeignPredicate(
    func=func,
    name=func_name,
    type_params=processed_type_params,
    input_arg_types=argument_types,
    output_arg_types=return_tuple_type,
    tag_type=return_tag_type,
    suppress_warning=suppress_warning,
  )


def _extract_facts_return_type(return_type):
  if get_origin(return_type) is not ClassVar:
    raise TypeError("Return type must be Facts")

  generator_args = get_args(return_type)

  if len(generator_args) != 1:
    raise TypeError("Return type must be Facts")

  generator_type = generator_args[0]
  if get_origin(generator_type) is not Generator:
    raise TypeError("Return type must be Facts")

  generator_type_args = get_args(generator_type)
  if len(generator_type_args) != 3:
    raise TypeError("Return type must be Facts")

  yield_type, send_type, generator_return_type = generator_type_args
  if not (_is_none_type(send_type) and _is_none_type(generator_return_type)):
    raise TypeError("Return type must be Facts")

  if get_origin(yield_type) is not Union:
    raise TypeError("Return type must be Facts")

  yield_type_args = get_args(yield_type)
  if len(yield_type_args) != 2:
    raise TypeError("Return type must be Facts")

  tagged_fact_type = yield_type_args[0]
  if get_origin(tagged_fact_type) is not tuple:
    raise TypeError("Return type must be facts")

  tagged_fact_args = get_args(tagged_fact_type)
  if len(tagged_fact_args) != 2:
    raise TypeError("Facts must have 2 type arguments")

  tag_type, tuple_type = tagged_fact_args
  return tag_types.get_tag_type(tag_type), _extract_return_tuple_type(tuple_type)


def _extract_return_tuple_type(tuple_type) -> List[Type]:
  # First check if it is a None type (i.e. returning zero-tuple)
  if _is_none_type(tuple_type):
    return []

  # Then try to convert it to a base type
  try:
    ty = Type(tuple_type)
    return [ty]
  except TypeError:
    pass

  # If not, it must be a tuple of base types
  if get_origin(tuple_type) is tuple:
    return [Type(t) for t in get_args(tuple_type)]
  else:
    raise TypeError("Return tuple type must be a base type or a tuple of base types")


def _is_none_type(ty):
  return ty is None or ty is NoneType
