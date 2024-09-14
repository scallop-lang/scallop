from typing import Optional

from . import AstNode, NodeLocation

class AstEnumNode(AstNode):
  def __init__(self, loc: NodeLocation, key: str, internal: Optional[AstNode]):
    self._loc = loc
    self._key = key
    self._internal = internal

  def loc(self) -> NodeLocation:
    return self._loc

  def __getattr__(self, attr_name: str):
    if attr_name[0:3] == "is_":
      attr = attr_name[3:]
      return lambda: self._key.lower() == attr
    if attr_name[0:3] == "as_":
      attr = attr_name[3:]
      if self._key.lower() == attr:
        return lambda: self._internal
      else:
        return lambda: None
    else:
      return super().__getattr__(attr_name)

  def __repr__(self):
    if self._internal is not None:
      return f"{self._key}({self._internal})"
    else:
      return f"{self._key}"
