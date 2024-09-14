from typing import Union, Dict, Any

from .. import value_types

from . import AstNode, NodeLocation

class AstTypeNode(AstNode):
  def __init__(self, loc: NodeLocation, type_name: str):
    self._loc = loc
    self._type_name = type_name

  def loc(self):
    return self._loc

  def name(self):
    return self._type_name

  def to_python_type(self):
    TYPE_MAPPING: Dict[str, Any] = {
      value_types.u8.__name__: int,
      value_types.u16.__name__: int,
      value_types.u32.__name__: int,
      value_types.u64.__name__: int,
      value_types.u128.__name__: int,
      value_types.usize.__name__: int,
      value_types.i8.__name__: int,
      value_types.i16.__name__: int,
      value_types.i32.__name__: int,
      value_types.i64.__name__: int,
      value_types.i128.__name__: int,
      value_types.isize.__name__: int,
      value_types.f32.__name__: float,
      value_types.f64.__name__: float,
      value_types.bool.__name__: bool,
      value_types.char.__name__: str,
      value_types.String.__name__: str,
      value_types.Symbol.__name__: str,
      value_types.DateTime.__name__: str,
      value_types.Duration.__name__: str,
      value_types.Entity.__name__: str,
    }
    if self._type_name not in TYPE_MAPPING:
      raise Exception(f"Unknown scallop type `{self._type_name}`; aborting")
    return TYPE_MAPPING[self._type_name]

  def parse_value(self, text: str) -> Union[float, int, bool, str]:
    ty = self.to_python_type()
    if ty == bool:
      if text == "true" or text == "True": return True
      elif text == "false" or text == "False": return False
      else: return False
    elif ty == int:
      return int(float(text))
    else:
      return ty(text)

  def __getattr__(self, attr_name: str):
    if attr_name[0:3] == "is_":
      attr = attr_name[3:]
      return lambda: self._type_name.lower() == attr
    else:
      raise Exception(f"Unknown attribute `{attr_name}`")

  def __repr__(self):
    return f"Type({self._type_name})"
