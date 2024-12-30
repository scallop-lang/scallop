from typing import List
import csv

import scallopy

from ..plugin import ScallopCodeQLPlugin

FIELDS = {
  "id": {
    "name": "id",
    "type": "usize",
    "retrieve": lambda id, row, _: id,
  },
  "node": {
    "name": "class_name",
    "type": "String",
    "retrieve": lambda id, row, _: row[0],
  },
  "file_name": {
    "name": "file_name",
    "type": "String",
    "retrieve": lambda id, row, _: row[1],
  },
  "start_line": {
    "name": "start_line",
    "type": "usize",
    "retrieve": lambda id, row, _: int(row[2]),
  },
  "end_line": {
    "name": "end_line",
    "type": "usize",
    "retrieve": lambda id, row, _: int(row[3]),
  },
  "enclosing_method_id": {
    "name": "enclosing_method_id",
    "type": "usize",
    "retrieve": lambda id, row, method_def: method_def.retrieve_id(row[4], row[1], int(row[5]), int(row[6])),
  }
}

ARG_CHOICES = list(FIELDS.keys())

ERR_HEAD = "[scallop_codeql/get_dataflow_node]"

def get_dataflow_node(arg_names: List[str], arg_types: List[scallopy.AstTypeNode], plugin: ScallopCodeQLPlugin, debug: bool = False):
  # Check the well-formedness of each argument
  for (arg_name, arg_type) in zip(arg_names, arg_types):
    assert arg_name in FIELDS, f"{ERR_HEAD} unknown argument `{arg_name}`. Choose from {ARG_CHOICES}"
    assert arg_type.name() == FIELDS[arg_name]["type"], f"{ERR_HEAD} expected `{FIELDS[arg_name]['type']}` for argument `{arg_name}`"

  # Generate the foreign predicate
  @scallopy.foreign_predicate(name="get_dataflow_node", input_arg_types=[], output_arg_types=arg_types, tag_type=None)
  def get_dataflow_node_impl():
    if plugin.is_java():
      for item in get_dataflow_node_java():
        yield item
    else:
      raise Exception(f"{ERR_HEAD} Unsupported for language `{plugin.project_language()}`")

  # Java version of get dataflow node
  def get_dataflow_node_java():
    # Make sure dataflow node are ready when we need it
    if "enclosing_method_id" in arg_names:
      method_def_csv_file_dir = plugin.run_codeql_query("java/queries/extract_method_definitions.ql", "method_definitions")
      method_def = MethodDefinition(method_def_csv_file_dir)
    else:
      method_def = None

    # Get facts
    csv_file_dir = plugin.run_codeql_query("java/queries/extract_dataflow_nodes.ql", "dataflow_nodes")
    for (row_id, row) in enumerate(csv.reader(open(csv_file_dir))):
      if row_id == 0: continue
      try:
        fact = tuple([FIELDS[field_name]["retrieve"](row_id, row, method_def) for field_name in arg_names])
        yield fact
      except:
        pass

  # Return the foreign predicate
  return get_dataflow_node_impl


class MethodDefinition:
  def __init__(self, csv_file_dir: str):
    self.mapping = {}
    for (row_id, row) in enumerate(csv.reader(open(csv_file_dir))):
      if row_id == 0: continue
      method_name, file_name, start_line, start_column = row[0], row[2], int(row[3]), int(row[4])
      self.mapping[(method_name, file_name, start_line, start_column)] = row_id

  def retrieve_id(self, method_name: str, file_name: str, start_line: int, start_column: int) -> int:
    return self.mapping[(method_name, file_name, start_line, start_column)]
