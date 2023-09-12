from typing import Dict, List, Union, Tuple, Optional, Any

from .provenance import ScallopProvenance
from .input_output import CSVFileOptions


def torch_tensor_enabled() -> bool:
  """
  Returns a boolean indicating whether this version of `scallopy`
  is compiled with torch tensor enabled.
  """


class InternalScallopContext:
  def __init__(
    self,
    provenance: str = "unit",
    k: int = 3,
    custom_provenance: Optional[ScallopProvenance] = None,
  ) -> None: ...

  def import_file(self, file_name: str): ...

  def clone(self) -> InternalScallopContext: ...

  def clone_with_new_provenance(
    self,
    provenance: str,
    custom_provenance: Any,
    k: int,
  ) -> InternalScallopContext: ...

  def set_non_incremental(self): ...

  def compile(self): ...

  def enable_tensor_registry(self): ...

  def set_k(self, k: int): ...

  def set_early_discard(self, early_discard: bool = True): ...

  def set_iter_limit(self, k: int): ...

  def remove_iter_limit(self): ...

  def run(self, iter_limit: Optional[int]) -> None: ...

  def run_with_debug_tag(self, iter_limit: Optional[int]) -> None: ...

  def run_batch(
    self,
    iter_limit: Optional[int],
    output_relation: str,
    inputs: Dict[str, List[Tuple[List, Optional[List[List[int]]]]]],
  ) -> List[InternalScallopCollection]: ...

  def add_relation(
    self,
    relation: str,
    load_csv: Optional[Union[CSVFileOptions, str]] = None,
  ) -> None: ...

  def add_facts(self, relation: str, elems: List[Tuple]) -> None: ...

  def check_tuple(self, relation: str, elem: Tuple) -> bool: ...

  def check_tuples(self, relation: str, elems: List[Tuple]) -> bool: ...

  def add_program(self, program: str) -> None: ...

  def add_item(self, program: str) -> List[str]: ...

  def add_rule(
    self,
    rule: str,
    tag: Optional[Any] = None,
    demand: Optional[str] = None,
  ) -> None: ...

  def compile_entity(self, relation: str, entity_tuple: Tuple[str]) -> Dict[str, List[Tuple]]: ...

  def register_foreign_function(self, ff: Any) -> None: ...

  def register_foreign_predicate(self, fp: Any) -> None: ...

  def register_foreign_attribute(self, ff: Any) -> None: ...

  def dump_front_ir(self): ...

  def relation(self, relation: str) -> InternalScallopCollection: ...

  def relation_with_debug_tag(self, relation: str) -> InternalScallopCollection: ...

  def has_relation(self, relation: str) -> bool: ...

  def relation_is_computed(self, relation: str) -> bool: ...

  def num_relations(self, include_hidden: bool = False) -> int: ...

  def relations(self, include_hidden: bool = False) -> List[str]: ...

  def relation_field_names(self, relation: str) -> List[Optional[str]]: ...


class InternalScallopCollection:
  """
  A collection of tuples (maybe associated with tags)
  """
  def num_input_facts(self) -> Optional[int]:
    """
    Get the number of input facts for a valid provenance semiring
    """

  def input_tags(self) -> Optional[List[Any]]:
    """
    Get all the input tags
    """

  def len(self) -> int:
    """
    Get the number of elements in the collection
    """

  def __iter__(self) -> InternalScallopCollectionIterator:
    """
    Iterate through the tuples of the collection
    """


class InternalScallopCollectionIterator:
  """
  An iterator over the scallop collection
  """

  def __next__(self) -> Tuple:
    """
    Get the next tuple in the collection
    """
