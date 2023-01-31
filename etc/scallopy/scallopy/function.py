from typing import *
import inspect


class GenericTypeParameter:
  """
  A generic type parameter used for Scallop foreign function
  """

  COUNTER = 0

  def __init__(self, type_family = "Any"):
    self.id = GenericTypeParameter.COUNTER
    if type_family == "Any" or type_family == any or type_family == None:
      self.type_family = "Any"
    elif type_family == "Number":
      self.type_family = "Number"
    elif type_family == "Integer" or type_family == int:
      self.type_family = "Integer"
    elif type_family == "SignedInteger":
      self.type_family = "SignedInteger"
    elif type_family == "UnsignedInteger":
      self.type_family = "UnsignedInteger"
    elif type_family == "Float" or type_family == float:
      self.type_family = "Float"
    else:
      raise Exception(f"Unknown type family {type_family}")
    GenericTypeParameter.COUNTER += 1

  def __repr__(self):
    return f"T{self.id}({self.type_family})"


class Type:
  def __init__(self, value):
    if type(value) == GenericTypeParameter:
      self.kind = "generic"
      self.id = value.id
      self.type_family = value.type_family
    elif value == "Float" or value == float:
      self.kind = "family"
      self.type_family = "Float"
    elif value == "Integer" or value == int:
      self.kind = "family"
      self.type_family = "Integer"
    elif value == "UnsignedInteger":
      self.kind = "family"
      self.type_family = "UnsignedInteger"
    elif value == "SignedInteger":
      self.kind = "family"
      self.type_family = "SignedInteger"
    elif value == "Number":
      self.kind = "family"
      self.type_family = "Number"
    elif value == "Any" or value == any or value == None:
      self.kind = "family"
      self.type_family = "Any"
    elif value == "char":
      self.kind = "base"
      self.type = "char"
    elif value == "bool" or value == bool:
      self.kind = "base"
      self.type = "bool"
    elif value == "String" or value == str:
      self.kind = "base"
      self.type = "String"
    elif value == "i8" or value == "i16" or value == "i32" or value == "i64" or value == "i128" or value == "isize" or \
      value == "u8" or value == "u16" or value == "u32" or value == "u64" or value == "u128" or value == "usize" or \
      value == "f32" or value == "f64":
      self.kind = "base"
      self.type = value
    else:
      raise Exception(f"Unknown scallop function type annotation")

  def __repr__(self):
    if self.kind == "base":
      return f"BaseType({self.type})"
    elif self.kind == "family":
      return f"TypeFamily({self.type_family})"
    elif self.kind == "generic":
      return f"Generic({self.id}, {self.type_family})"
    else:
      raise Exception(f"Unknown parameter kind {self.kind}")

  def is_generic(self):
    return self.kind == "generic"

  def is_type_family(self):
    return self.kind == "family"


class ForeignFunction:
  """
  A Scallop Foreign Function
  """
  def __init__(
    self,
    func: Callable,
    name: str,
    generic_type_params: List[str],
    static_arg_types: List[Type],
    optional_arg_types: List[Type],
    var_arg_types: Optional[Type],
    return_type: Type,
  ):
    self.func = func
    self.name = name
    self.generic_type_params = generic_type_params
    self.static_arg_types = static_arg_types
    self.optional_arg_types = optional_arg_types
    self.var_arg_types = var_arg_types
    self.return_type = return_type

  def __call__(self, *args):
    return self.func(*args)

  def arg_type_repr(self, arg):
    if arg.is_generic():
      return f"T{arg.id}"
    elif arg.is_type_family():
      return arg.type_family
    else:
      return arg.type

  def __repr__(self):
    r = f"extern fn ${self.name}"

    # Generic Type Parameters
    if len(self.generic_type_params) > 0:
      r += "<"
      for (i, param) in enumerate(self.generic_type_params):
        if i > 0:
          r += ", "
        r += f"T{i}: {param}"
      r += ">"

    # Start
    r += "("

    # Static arguments
    for (i, arg) in enumerate(self.static_arg_types):
      if i > 0:
        r += ", "
      r += self.arg_type_repr(arg)

    # Optional arguments
    if len(self.static_arg_types) > 0 and len(self.optional_arg_types) > 0:
      r += ", "
    for (i, arg) in enumerate(self.optional_arg_types):
      if i > 0:
        r += ", "
      r += f"{self.arg_type_repr(arg)}?"

    # Variable arguments
    if self.var_arg_types is not None:
      if len(self.static_arg_types) + len(self.optional_arg_types) > 0:
        r += ", "
      r += f"{self.arg_type_repr(self.var_arg_types)}..."

    # Return type
    r += f") -> {self.arg_type_repr(self.return_type)}"

    return r


def foreign_function(func):
  """
  A decorator to create a Scallop foreign function, for example

  ``` python
  @scallop_function
  def string_index_of(s1: str, s2: str) -> usize:
    return s1.index(s2)
  ```

  This foreign function can be then registered into Scallop for invokation

  ``` python
  ctx.register_foreign_function(string_index_of)
  ```
  """

  # Get the function name
  func_name = func.__name__

  # Get the function signature
  signature = inspect.signature(func)

  # Store all the argument types
  static_argument_types = []
  optional_argument_types = []
  variable_argument_type = None

  # Find argument types
  for (arg_name, item) in signature.parameters.items():
    optional = item.default != inspect.Parameter.empty
    if item.annotation is None:
      raise Exception(f"Argument {arg_name} type annotation not provided")
    if item.kind == inspect.Parameter.VAR_POSITIONAL:
      variable_argument_type = Type(item.annotation)
    elif not optional:
      static_argument_types.append(Type(item.annotation))
    else:
      if item.default != None:
        raise Exception("Optional arguments need to have default `None`")
      optional_argument_types.append(Type(item.annotation))

  # Get all argument types
  all_arg_types = static_argument_types + \
                  optional_argument_types + \
                  ([variable_argument_type] if variable_argument_type is not None else [])

  # Find return types
  if signature.return_annotation is None:
    raise Exception(f"Return type annotation not provided")
  return_type = Type(signature.return_annotation)

  # If the return type is generic, at least one of its argument also needs to have the same type
  if return_type.is_generic():
    is_return_generic_type_ok = False
    for arg_type in all_arg_types:
      if arg_type.is_generic() and arg_type.id == return_type.id:
        is_return_generic_type_ok = True
    if not is_return_generic_type_ok:
      raise Exception(f"Return generic type not bounded by any input argument")
  elif return_type.is_type_family():
    raise Exception(f"Return type cannot be a type family ({return_type})")

  # Put all types together and find generic type
  generic_types_map = {}
  generic_type_params = []
  all_types = all_arg_types + [return_type]
  for param in all_types:
    if param.is_generic():
      if param.id not in generic_types_map:
        generic_types_map[param.id] = []
      generic_types_map[param.id].append(param)
  for (i, (_, params)) in enumerate(generic_types_map.items()):
    assert len(params) > 0, "Should not happen; there has to be at least one type using generic type parameter"
    for param in params:
      param.id = i
    generic_type_params.append(params[0].type_family)

  # Return a Scallop Foreign Function class
  return ForeignFunction(
    func,
    func_name,
    generic_type_params,
    static_argument_types,
    optional_argument_types,
    variable_argument_type,
    return_type,
  )
