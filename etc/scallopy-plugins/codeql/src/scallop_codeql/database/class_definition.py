from typing import List
import csv

import scallopy

from ..plugin import ScallopCodeQLPlugin

FIELDS = {
  "id": {
    "name": "id",
    "type": "usize",
    "retrieve": lambda id, row: id,
  },
  "class_name": {
    "name": "class_name",
    "type": "String",
    "retrieve": lambda id, row: row[0],
  },
  "file_name": {
    "name": "file_name",
    "type": "String",
    "retrieve": lambda id, row: row[1],
  },
  "start_line": {
    "name": "start_line",
    "type": "usize",
    "retrieve": lambda id, row: int(row[2]),
  },
  "end_line": {
    "name": "end_line",
    "type": "usize",
    "retrieve": lambda id, row: int(row[3]),
  },
}

ARG_CHOICES = list(FIELDS.keys())

ERR_HEAD = "[scallop_codeql/get_class_definition]"

def get_class_definition(arg_names: List[str], arg_types: List[scallopy.AstTypeNode], plugin: ScallopCodeQLPlugin, debug: bool = False):
  # Check the well-formedness of each argument
  for (arg_name, arg_type) in zip(arg_names, arg_types):
    assert arg_name in FIELDS, f"{ERR_HEAD} unknown argument `{arg_name}`. Choose from {ARG_CHOICES}"
    assert arg_type.name() == FIELDS[arg_name]["type"], f"{ERR_HEAD} expected `{FIELDS[arg_name]['type']}` for argument `{arg_name}`"

  # Generate the foreign predicate
  @scallopy.foreign_predicate(name="get_class_definition", input_arg_types=[], output_arg_types=arg_types, tag_type=None)
  def get_class_definition_impl():
    if plugin.is_java():
      for item in get_class_definition_java():
        yield item
    else:
      raise Exception(f"{ERR_HEAD} Unsupported for language `{plugin.project_language()}`")

  # Java version of get class definition
  def get_class_definition_java():
    csv_file_dir = plugin.run_codeql_query("java/queries/extract_class_definitions.ql", "class_definitions")
    for (row_id, row) in enumerate(csv.reader(open(csv_file_dir))):
      if row_id == 0: continue
      fact = tuple([FIELDS[field_name]["retrieve"](row_id, row) for field_name in arg_names])
      yield fact

  # Return the foreign predicate
  return get_class_definition_impl
