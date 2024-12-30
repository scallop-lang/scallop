import scallopy

from ..plugin import ScallopCodeQLPlugin

FA_NAME = "codeql_database"
ERR_HEAD = f"[@{FA_NAME}]"

def get_codeql_database(plugin: ScallopCodeQLPlugin):
  @scallopy.foreign_attribute
  def codeql_database(item: scallopy.AstNode, *, debug: bool = False):
    """
    The foreign predicate that extracts a CodeQL database into Scallop relations
    """
    # Check if the attribute is used on relation type declarations
    assert item.is_relation_decl(), f"{ERR_HEAD} has to be an attribute of a relation type declaration"

    # For each relation declaration, generate a foreign predicate
    result_fps = [get_database_fp(*get_rel_decl_info(r), debug) for r in item.relation_decls()]

    # Return all the foreign predicates
    return result_fps

  def get_rel_decl_info(rel_decl: scallopy.AstNode):
    """
    Get the information of a single relation declaration
    """
    rel_name = rel_decl.name.name
    arg_names = [ab.name.name for ab in rel_decl.arg_bindings]
    arg_types = [ab.ty for ab in rel_decl.arg_bindings]
    assert all([ab.adornment == None or ab.adornment.is_free() for ab in rel_decl.arg_bindings]), f"{ERR_HEAD} all arguments of `{rel_name}` need to be free"
    return (rel_name, arg_names, arg_types)

  def get_database_fp(rel_name, arg_names, arg_types, debug):
    from .. import database
    DATABASE_FPS = {
      "get_class_definition": database.class_definition.get_class_definition,
      "get_method_definition": database.method_definition.get_method_definition,
      "get_local_dataflow_edge": database.local_dataflow_edge.get_local_dataflow_edge,
      "get_dataflow_node": database.dataflow_node.get_dataflow_node,
    }
    assert rel_name in DATABASE_FPS, f"{ERR_HEAD} unknown CodeQL database predicate `{rel_name}`"
    return DATABASE_FPS[rel_name](arg_names, arg_types, plugin, debug=debug)

  return codeql_database
