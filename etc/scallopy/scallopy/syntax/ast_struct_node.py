from typing import Dict

from . import AstNode, NodeLocation, AstConstantNode

class AstStructNode(AstNode):
  def __init__(self, loc: NodeLocation, fields: Dict[str, AstNode]):
    self._loc = loc
    self._fields = fields

  def loc(self) -> NodeLocation:
    return self._loc

  def __getattr__(self, attr_name: str):
    if attr_name in self._fields:
      obj = self._fields[attr_name]
      if isinstance(obj, AstConstantNode):
        value = obj.get()
        if isinstance(value, str): return value
        elif isinstance(value, bool): return value
        elif isinstance(value, int): return value
        elif isinstance(value, float): return value
      else:
        return obj
    else:
      return super().__getattr__(attr_name)

  def __repr__(self):
    fields = ", ".join([f"{k}: {v}" for k, v in self._fields.items()])
    return f"{{{fields}}}"
