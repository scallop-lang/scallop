from typing import Optional

from . import NodeLocation

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
    from . import AstTypeNode, AstVariantNode, AstEnumNode, AstStructNode, AstConstantNode

    if type(internal_item) == dict:
      keys = list(internal_item.keys())

      if len(keys) == 1:
        # Is a variant
        key = keys[0]
        internal = AstNode.parse(internal_item[key])
        return AstVariantNode(key, internal)

      elif "_loc" in keys and "_node" in keys:
        loc = NodeLocation(internal_item["_loc"])
        internal_node = internal_item["_node"]
        if type(internal_node) == str:
          # Is a enum node
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
