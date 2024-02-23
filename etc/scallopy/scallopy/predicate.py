import typing
from typing import TypeVar, Generic, Union, Tuple, Callable, List, Optional, Any, ForwardRef, ClassVar
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
      raise Exception(f"Unknown scallop predicate type annotation `{value}`")

  def __repr__(self):
    return self.type

  @staticmethod
  def sanitize_type_str(value):
    if value == "i8" or value == "i16" or value == "i32" or value == "i64" or value == "i128" or value == "isize" or \
      value == "u8" or value == "u16" or value == "u32" or value == "u64" or value == "u128" or value == "usize" or \
      value == "f32" or value == "f64" or \
      value == "bool" or value == "char" or value == "String" or value == "Symbol" or value == "Tensor" or \
      value == "DateTime" or value == "Duration" or value == "Entity":
      return value
    elif value in ALIASES:
      return ALIASES[value]
    else:
      raise Exception(f"Unknown scallop predicate type annotation `{value}`")


TagType = TypeVar("TagType")

TupleType = TypeVar("TupleType")

Facts = ClassVar[typing.Generator[
  Union[
    Tuple[TagType, TupleType],
    TupleType,
  ],
  None,
  None,
]]

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
      for (i, type_param) in enumerate(self.type_params):
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

  # Store all the type params
  if type_params is None:
    processed_type_params = []
  else:
    processed_type_params = [Type(type_param) for type_param in type_params]

  # Store all the argument types
  if input_arg_types is None:
    argument_types = []

    # Find argument types
    for (arg_name, item) in signature.parameters.items():
      optional = item.default != inspect.Parameter.empty
      if item.annotation is None:
        raise Exception(f"Argument {arg_name} type annotation not provided")
      if item.kind == inspect.Parameter.VAR_POSITIONAL:
        raise Exception(f"Cannot have variable arguments in foreign predicate")
      elif not optional:
        ty = Type(item.annotation)
        argument_types.append(ty)
      else:
        raise Exception(f"Cannot have optional argument in foreign predicate")
  else:
    argument_types = [Type(t) for t in input_arg_types]

  # Find return type
  if output_arg_types is None:
    if signature.return_annotation is None:
      raise Exception(f"Return type annotation not provided")
    elif not str(signature.return_annotation).startswith("typing.ClassVar[typing.Generator[typing.Union[typing.Tuple["):
      raise Exception(f"Return type must be Facts")
    else:
      args = signature.return_annotation \
        .__dict__["__args__"][0] \
        .__dict__["__args__"][0] \
        .__dict__["__args__"][0] \
        .__dict__["__args__"]
      if len(args) != 2:
        raise Exception(f"Facts must have 2 type arguments")

      # Produce return tag type
      return_tag_type = tag_types.get_tag_type(args[0])

      # Produce return tuple type, and check that they are all base type
      return_tuple_type = _extract_return_tuple_type(args[1])
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


def _extract_return_tuple_type(tuple_type) -> List[Type]:
  # First check if it is a None type (i.e. returning zero-tuple)
  if tuple_type == None:
    return []

  # Then try to convert it to a base type
  try:
    ty = Type(tuple_type)
    return [ty]
  except: pass

  # If not, it must be a tuple of base types
  if "__origin__" in tuple_type.__dict__ and tuple_type.__dict__["__origin__"] == tuple:
    if "__args__" in tuple_type.__dict__:
      return [Type(t) for t in tuple_type.__dict__["__args__"]]
    else:
      return []
  else:
    raise Exception(f"Return tuple type must be a base type or a tuple of base types")
