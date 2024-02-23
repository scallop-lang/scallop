from typing import Optional, Union, Dict, List, Any

from . import value_types


class NodeLocation:
  def __init__(self, internal_location):
    self._offset_span = (internal_location["offset_span"]["start"], internal_location["offset_span"]["end"])
    self._id = internal_location["id"]
    self._souce_id = internal_location["source_id"]
    if internal_location["loc_span"]:
      self._loc_span = (
        (internal_location["loc_span"]["start"]["row"], internal_location["loc_span"]["start"]["col"]),
        (internal_location["loc_span"]["end"]["row"], internal_location["loc_span"]["end"]["col"]))
    else:
      self._loc_span = None

  def __repr__(self):
    if self._id is not None:
      if self._loc_span is not None:
        return f"[#{self._id} {self._loc_span[0][0]}:{self._loc_span[0][1]} - {self._loc_span[1][0]}:{self._loc_span[1][1]}]"
      else:
        return f"[#{self._id} {self._offset_span[0]}-{self._offset_span[1]}]"
    else:
      if self._loc_span is not None:
        return f"[{self._loc_span[0][0]}:{self._loc_span[0][1]} - {self._loc_span[1][0]}:{self._loc_span[1][1]}]"
      else:
        return f"[{self._offset_span[0]}-{self._offset_span[1]}]"


class AstNode:
  def loc(self) -> Optional[NodeLocation]:
    return None

  def __getattr__(self, attr_name: str):
    if attr_name[0:3] == "is_":
      return False
    else:
      raise Exception(f"Unknown attribute `{attr_name}`")

  @staticmethod
  def parse(internal_item) -> "AstNode":
    if type(internal_item) == dict:
      keys = list(internal_item.keys())
      if len(keys) == 1:
        key = keys[0]
        internal = AstNode.parse(internal_item[key])
        return AstVariantNode(key, internal)
      elif "_loc" in keys and "_node" in keys:
        loc = NodeLocation(internal_item["_loc"])
        internal_node = internal_item["_node"]
        if type(internal_node) == str:
          return AstEnumNode(loc, internal_node, None)
        else: # can only be dictionary
          node_keys = list(internal_item["_node"])
          if len(node_keys) == 1 and node_keys[0][0].isupper():
            return AstEnumNode(loc, internal_node, AstNode.parse(internal_item["_node"][node_keys[0]]))
          elif "__kind__" in node_keys and internal_item["_node"]["__kind__"] == "type":
            return AstTypeNode(loc, internal_item["_node"]["type"])
          else:
            fields = {k: AstNode.parse(v) for k, v in internal_item["_node"].items()}
            return AstStructNode(loc, fields)
      else:
        raise Exception(f"Unknown object {internal_item}")
    elif type(internal_item) == list:
      return [AstNode.parse(item) for item in internal_item]
    else:
      return AstConstantNode(internal_item)


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


class AstTypeNode(AstNode):
  def __init__(self, loc: NodeLocation, type_name: str):
    self._loc = loc
    self._type_name = type_name

  def loc(self):
    return self._loc

  def name(self):
    return self._type_name

  def to_python_type(self):
    return {
      value_types.u8: int,
      value_types.u16: int,
      value_types.u32: int,
      value_types.u64: int,
      value_types.u128: int,
      value_types.usize: int,
      value_types.i8: int,
      value_types.i16: int,
      value_types.i32: int,
      value_types.i64: int,
      value_types.i128: int,
      value_types.isize: int,
      value_types.f32: float,
      value_types.f64: float,
      value_types.bool: bool,
      value_types.char: str,
      value_types.String: str,
      value_types.Symbol: str,
      value_types.DateTime: str,
      value_types.Duration: str,
      value_types.Entity: str,
    }[self._type_name]

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
