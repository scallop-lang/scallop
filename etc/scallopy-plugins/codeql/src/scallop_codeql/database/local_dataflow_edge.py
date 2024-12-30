from typing import List
import csv
from collections import defaultdict

import scallopy

from ..plugin import ScallopCodeQLPlugin

FIELDS = {
  "source_id": {
    "name": "source_id",
    "type": "usize",
    "retrieve": lambda id, row: int(row[0]),
  },
  "sink_id": {
    "name": "sink_id",
    "type": "usize",
    "retrieve": lambda id, row: int(row[1]),
  },
}

ARG_CHOICES = list(FIELDS.keys())

ERR_HEAD = "[scallop_codeql/get_local_dataflow_edge]"

def get_local_dataflow_edge(arg_names: List[str], arg_types: List[scallopy.AstTypeNode], plugin: ScallopCodeQLPlugin, debug: bool = False):
  # Check the well-formedness of each argument
  for (arg_name, arg_type) in zip(arg_names, arg_types):
    assert arg_name in FIELDS, f"{ERR_HEAD} unknown argument `{arg_name}`. Choose from {ARG_CHOICES}"
    assert arg_type.name() == FIELDS[arg_name]["type"], f"{ERR_HEAD} expected `{FIELDS[arg_name]['type']}` for argument `{arg_name}`"

  # Generate the foreign predicate
  @scallopy.foreign_predicate(name="get_local_dataflow_edge", input_arg_types=[], output_arg_types=arg_types, tag_type=None)
  def get_local_dataflow_edge_impl():
    if plugin.is_java():
      for item in get_local_dataflow_edge_java():
        yield item
    else:
      raise Exception(f"{ERR_HEAD} Unsupported for language `{plugin.project_language()}`")

  # Java version of get method definition
  def get_local_dataflow_edge_java():
    if debug: print(f"{ERR_HEAD} extracting dataflow nodes...")
    dataflow_nodes_csv_dir = plugin.run_codeql_query("java/queries/extract_dataflow_nodes.ql", "dataflow_nodes", debug=debug)
    dataflow_nodes_id_map = DataFlowNodesIdMap(dataflow_nodes_csv_dir)
    if debug: print(f"{ERR_HEAD} extracting dataflow edges...")
    dataflow_edges_csv_dir = plugin.run_codeql_query("java/queries/extract_local_dataflow_edges.ql", "local_dataflow_edges", debug=debug)
    if debug: print(f"{ERR_HEAD} caching dataflow indices...")
    dataflow_edge_indices_csv_dir = create_and_cache_dataflow_edge_indices(dataflow_nodes_id_map, dataflow_edges_csv_dir)
    if debug: print(f"{ERR_HEAD} fetching")
    for (row_id, row) in enumerate(csv.reader(open(dataflow_edge_indices_csv_dir))):
      if row_id == 0: continue
      try:
        yield tuple([FIELDS[field_name]["retrieve"](row_id, row) for field_name in arg_names])
      except:
        continue

  def create_and_cache_dataflow_edge_indices(dataflow_nodes_id_map: DataFlowNodesIdMap, dataflow_edges_csv_dir: str):
    DATAFLOW_EDGE_INDICES_NAME = "dataflow_edge_indices"
    maybe_indices = plugin.try_load_index_csv(DATAFLOW_EDGE_INDICES_NAME)
    if maybe_indices is not None:
      return maybe_indices
    else:
      dataflow_edge_indices = []
      for (row_id, row) in enumerate(csv.reader(open(dataflow_edges_csv_dir))):
        if row_id == 0: continue
        file_name = row[6]
        source = (row[0], int(row[1]), int(row[2]))
        sink = (row[3], int(row[4]), int(row[5]))
        try:
          source_id = dataflow_nodes_id_map.retrieve_id(file_name, *source)
          sink_id = dataflow_nodes_id_map.retrieve_id(file_name, *sink)
          if source_id == sink_id: continue
          dataflow_edge_indices.append((source_id, sink_id))
        except:
          pass
      return plugin.save_index_csv(DATAFLOW_EDGE_INDICES_NAME, ["source_id", "sink_id"], dataflow_edge_indices)

  # Return the foreign predicate
  return get_local_dataflow_edge_impl


class DataFlowNodesIdMap:
  def __init__(self, csv_dir: str):
    self.mapping = defaultdict(lambda: dict())
    for (row_id, row) in enumerate(csv.reader(open(csv_dir))):
      if row_id == 0: continue
      node = row[0]
      file_name = row[1]
      start_line = int(row[2])
      start_column = int(row[3])
      self.mapping[file_name][(node, start_line, start_column)] = row_id

  def retrieve_id(self, file_name, node, start_line, start_column):
    return self.mapping[file_name][(node, start_line, start_column)]
