from typing import List

from . import AstNode, NodeLocation, AstTypeNode

class AstVariantNode(AstNode):
  def __init__(self, key: str, internal: AstNode):
    self._key = key
    self._internal = internal

  def key(self) -> str:
    return self._key

  def loc(self) -> NodeLocation:
    return self._internal.loc()

  def is_function_decl(self) -> bool:
    return self.is_typedecl() and self.as_typedecl().is_function()

  def function_decl_name(self) -> str:
    """Assuming this node is an `Item`, get the function declaration's name"""
    return self.as_typedecl().as_function().func_name.name

  def function_decl_arg_types(self) -> List[AstTypeNode]:
    """Assuming this node is an `Item`, get the function declaration's argument types"""
    return [arg.ty for arg in self.as_typedecl().as_function().args]

  def function_decl_ret_type(self) -> AstTypeNode:
    """Assuming this node is an `Item`, get the function declaration's return type"""
    return self.as_typedecl().as_function().ret_ty

  def is_relation_decl(self) -> bool:
    return self.is_typedecl() and self.as_typedecl().is_relation()

  def relation_decls(self) -> List[AstNode]:
    return self.as_typedecl().as_relation().rel_types

  def relation_decl(self, i: int) -> AstNode:
    return self.as_typedecl().as_relation().rel_types[i]

  def __getattr__(self, attr_name: str):
    if attr_name[0:3] == "is_":
      attr = attr_name[3:]
      return lambda: self._key.lower() == attr
    if attr_name[0:3] == "as_":
      attr = attr_name[3:]
      if self._key.lower() == attr:
        return lambda: self._internal
      else:
        return None
    else:
      return super().__getattr__(attr_name)

  def __repr__(self):
    return f"{self._key}({self._internal})"
