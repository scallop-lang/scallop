from typing import Any

from . import AstNode

class AstConstantNode(AstNode):
  def __init__(self, value: Any):
    self._value = value

  def get(self) -> Any:
    return self._value

  def __repr__(self):
    if type(self._value) == str:
      return f"\"{self._value}\""
    else:
      return str(self._value)
