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
  "method_name": {
    "name": "function_name",
    "type": "String",
    "retrieve": lambda id, row, _: row[0],
  },
  "class_id": {
    "name": "class_id",
    "type": "usize",
    "retrieve": lambda id, row, class_def: class_def.get(row[2], row[1]),
  },
  "class_name": {
    "name": "class_name",
    "type": "String",
    "retrieve": lambda id, row, _: row[1],
  },
  "file_name": {
    "name": "file_name",
    "type": "String",
    "retrieve": lambda id, row, _: row[2],
  },
  "start_line": {
    "name": "start_line",
    "type": "usize",
    "retrieve": lambda id, row, _: int(row[3]),
  },
  "end_line": {
    "name": "end_line",
    "type": "usize",
    "retrieve": lambda id, row, _: int(row[4]),
  },
}

ARG_CHOICES = list(FIELDS.keys())

ERR_HEAD = "[scallop_codeql/get_method_definition]"

def get_method_definition(arg_names: List[str], arg_types: List[scallopy.AstTypeNode], plugin: ScallopCodeQLPlugin, debug: bool = False):
  # Check the well-formedness of each argument
  for (arg_name, arg_type) in zip(arg_names, arg_types):
    assert arg_name in FIELDS, f"{ERR_HEAD} unknown argument `{arg_name}`. Choose from {ARG_CHOICES}"
    assert arg_type.name() == FIELDS[arg_name]["type"], f"{ERR_HEAD} expected `{FIELDS[arg_name]['type']}` for argument `{arg_name}`"

  # Generate the foreign predicate
  @scallopy.foreign_predicate(name="get_method_definition", input_arg_types=[], output_arg_types=arg_types, tag_type=None)
  def get_method_definition_impl():
    if plugin.is_java():
      for item in get_method_definition_java():
        yield item
    else:
      raise Exception(f"{ERR_HEAD} Unsupported for language `{plugin.project_language()}`")

  # Java version of get method definition
  def get_method_definition_java():
    # Make sure class definitions are ready when we need it
    if "class_id" in arg_names:
      class_def_csv_file_dir = plugin.run_codeql_query("java/queries/extract_class_definitions.ql", "class_definitions")
      class_def = ClassDefinition(class_def_csv_file_dir)
    else:
      class_def = None

    # Load method definitions
    csv_file_dir = plugin.run_codeql_query("java/queries/extract_method_definitions.ql", "method_definitions")
    for (row_id, row) in enumerate(csv.reader(open(csv_file_dir))):
      if row_id == 0: continue
      try:
        fact = tuple([FIELDS[field_name]["retrieve"](row_id, row, class_def) for field_name in arg_names])
        yield fact
      except:
        continue

  # Return the foreign predicate
  return get_method_definition_impl


class ClassDefinition:
  def __init__(self, csv_file_dir: str):
    self.mapping = {}
    for (row_id, row) in enumerate(csv.reader(open(csv_file_dir))):
      if row_id == 0: continue
      class_name, file_name = row[0], row[1]
      self.mapping[(file_name, class_name)] = row_id

  def get(self, file_name: str, class_name: str) -> int:
    return self.mapping[(file_name, class_name)]
