from typing import *
import inspect


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
      value = value.__forward_arg__
    if value == float:
      self.type = "f32"
    elif value == int:
      self.type = "i32"
    elif value == bool:
      self.type = "bool"
    elif value == str:
      self.type = "String"
    elif value == "i8" or value == "i16" or value == "i32" or value == "i64" or value == "i128" or value == "isize" or \
      value == "u8" or value == "u16" or value == "u32" or value == "u64" or value == "u128" or value == "usize" or \
      value == "f32" or value == "f64" or \
      value == "bool" or value == "char" or value == "String" or \
      value == "DateTime" or value == "Duration":
      self.type = value
    elif value in ALIASES:
      self.type = ALIASES[value]
    else:
      raise Exception(f"Unknown scallop predicate type annotation `{value}`")

  def __repr__(self):
    return self.type


class Generator(Generic[TypeVar("TagType"), TypeVar("TupleType")]):
  pass


class ForeignPredicate:
  """
  Scallop foreign predicate
  """
  def __init__(
    self,
    func: Callable,
    name: str,
    input_arg_types: List[Type],
    output_arg_types: List[Type],
    tag_type: Any,
  ):
    self.func = func
    self.name = name
    self.input_arg_types = input_arg_types
    self.output_arg_types = output_arg_types
    self.tag_type = tag_type

  def __repr__(self):
    r = f"extern pred {self.name}[{self.pattern()}]("
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
    return self.tag_type is not None and self.tag_type is not type(None)


def foreign_predicate(func: Callable):
  """
  A decorator to create a Scallop foreign predicate, for example

  ``` python
  @scallopy.foreign_function
  def string_chars(s: str) -> scallopy.Generator[Tuple[int, char]]:
    for (i, c) in enumerate(s):
      yield (i, c)
  ```
  """

  # Get the function name
  func_name = func.__name__

  # Get the function signature
  signature = inspect.signature(func)

  # Store all the argument types
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

  # Find return type
  if signature.return_annotation is None:
    raise Exception(f"Return type annotation not provided")
  elif signature.return_annotation.__dict__["__origin__"] != Generator:
    raise Exception(f"Return type must be Generator")
  else:
    args = signature.return_annotation.__dict__["__args__"]
    if len(args) != 2:
      raise Exception(f"Generator must have 2 type arguments")

    # Produce return tag type
    return_tag_type = _extract_return_tag_type(args[0])

    # Produce return tuple type, and check that they are all base type
    return_tuple_type = _extract_return_tuple_type(args[1])


  # Create the foreign predicate
  return ForeignPredicate(
    func=func,
    name=func_name,
    input_arg_types=argument_types,
    output_arg_types=return_tuple_type,
    tag_type=return_tag_type,
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


def _extract_return_tag_type(tag_type):
  return tag_type
