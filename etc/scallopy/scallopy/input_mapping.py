from typing import *
from functools import reduce
import itertools
from copy import deepcopy

from . import torch_importer
from .utils import Counter

class InputMapping:
  """
  An input mapping for converting tensors into probabilistic symbols.

  Input mappings in Scallop can be constructed by many different ways,
  such as through single values, tuples, lists, iterators, and
  dictionaries.
  """

  def __init__(
    self,
    mapping,
    disjunctive: bool = False,
    disjunctive_dim: Optional[int] = None,
    retain_threshold: Optional[float] = None,
    retain_k: Optional[int] = None,
    sample_dim: Optional[int] = None,
    sample_strategy: Optional[Literal["top", "categorical"]] = "top",
    supports_disjunctions: bool = False,
  ):
    """Create a new input mapping"""

    # Initialize the mapping information
    ty = type(mapping)

    # First check if we are directly copying from another InputMapping
    if ty == InputMapping:
      self.__dict__ = deepcopy(mapping.__dict__)
      self.supports_disjunctions = supports_disjunctions
      return

    # Otherwise, process from beginning
    if ty == list or ty == range:
      self._initialize_list_mapping(mapping)
    elif ty == dict:
      self._initialize_dict_mapping(mapping)
    elif ty == tuple:
      self._initialize_tuple_mapping(mapping)
    elif self._is_primitive(mapping):
      self._initialize_value_mapping(mapping)
    elif mapping == None:
      self._kind = None
    else:
      raise Exception(f"Unknown input mapping type `{type(mapping)}`")

    # Initialize the properties
    self.disjunctive = disjunctive or (disjunctive_dim is not None)
    self.disjunctive_dim = disjunctive_dim
    self.retain_threshold = retain_threshold
    self.retain_k = retain_k
    self.sample_dim = sample_dim
    self.sample_strategy = sample_strategy
    self.supports_disjunctions = supports_disjunctions

    # Validate the configurations
    if self.disjunctive_dim is not None:
      if not (0 <= self.disjunctive_dim < self.dimension):
        raise Exception(f"Invalid disjunction dimension {self.disjunctive_dim}; total dimension is {self.dimension}")
    if self.sample_dim is not None:
      if not (0 <= self.sample_dim < self.dimension):
        raise Exception(f"Invalid sampling dimension {self.sample_dim}; total dimension is {self.dimension}")

  def set_sample_topk_facts(self, amount: int):
    self.retain_k = amount
    self.sample_dim = None
    self.sample_strategy = "top"

  def __getitem__(self, index) -> Tuple:
    """Get the tuple of the input mapping from an index"""
    if self._kind == "dict":
      return tuple([self._mapping[i][j] for (i, j) in enumerate(index)])
    else:
      return self._mapping[self._mult_dim_index_to_index(index)]

  def all_tuples(self) -> Iterator[Tuple]:
    """Iterate over all the tuples in the input mapping"""
    if self._kind == "list" or self._kind == "tuple" or self.kind == "value":
      for element in self._mapping:
        yield element
    elif self.kind == "dict":
      for element in itertools.product(*self._mapping):
        yield element

  def all_indices(self) -> Iterator[Tuple]:
    return itertools.product(*[list(range(x)) for x in self.shape])

  def process_tensor(self, tensor: torch_importer.Tensor, batched=False, mutual_exclusion_counter=None) -> List:
    """Process a tensor to produce a list of probabilistic symbols"""

    # Check the kind, if there is none
    if self._kind == None:
      raise Exception("Cannot apply None mapping to a tensor")

    # Create a new mutual exclusion counter if needed
    if self.supports_disjunctions and mutual_exclusion_counter is None:
      mutual_exclusion_counter = Counter()

    # Check the shape to decide whether to process a batched input
    if tensor.shape == self._shape:
      facts = self._process_one_tensor(tensor, mutual_exclusion_counter)
      return [facts] if batched else facts
    elif tensor.shape[1:] == self._shape:
      return [self._process_one_tensor(item, mutual_exclusion_counter) for item in tensor]
    else:
      raise Exception(f"Tensor shape mismatch: expected {self._shape}, got {tensor.shape}")

  def _process_one_tensor(self, tensor: torch_importer.Tensor, mutual_exclusion_counter: Counter) -> List:
    inc, exc = InclusiveSet(), ExclusiveSet()

    # Do sampling
    import torch
    if self.retain_k is not None:
      if self.sample_dim is not None:
        if self.sample_strategy == "categorical":
          transposed_tensor = tensor.transpose(self.sample_dim, tensor.dim() - 1)
          distributions = torch.distributions.Categorical(transposed_tensor)
          sampled_indices = distributions.sample((self.retain_k,)).transpose(0, transposed_tensor.dim() - 1)
          for index in self._convert_categorical_sampled_indices(sampled_indices, self.sample_dim):
            inc.add(index)
        elif self.sample_strategy == "top":
          topk_result = torch.topk(tensor, self.retain_k, dim=self.sample_dim)
          for index in self._convert_topk_sampled_indices(topk_result.indices, self.sample_dim):
            inc.add(index)
        else:
          raise Exception(f"Unknown sample strategy `{self.sample_strategy}`")
      else:
        flat_tensor = torch.flatten(tensor)
        if self.sample_strategy == "categorical":
          categorical_distr = torch.distributions.Categorical(probs=flat_tensor)
          sampled_indices = categorical_distr.sample((self.retain_k,))
          for index in sampled_indices:
            inc.add(self._index_to_mult_dim_index(int(index)))
        elif self.sample_strategy == "top":
          topk_result = torch.topk(flat_tensor, self.retain_k)
          for index in topk_result.indices:
            inc.add(self._index_to_mult_dim_index(int(index)))
        else:
          raise Exception(f"Unknown sample strategy `{self.sample_strategy}`")

    # Do thresholding; if the probability is less than the threshold, add the index to the exclusive map
    if self.retain_threshold is not None:
      for index in self.all_indices():
        if tensor[index] < self.retain_threshold:
          exc.exclude(index)

    # Get a set of filtered indices
    filtered_indices = [index for index in self.all_indices() if inc.contains(index) and exc.contains(index)]

    # Add disjunctions
    if self.supports_disjunctions:
      if self.disjunctive:
        if self.disjunctive_dim is not None:
          partial_indices = itertools.product(*[range(d) for (i, d) in enumerate(self.shape) if i != self.disjunctive_dim])
          disj_map = {index: mutual_exclusion_counter.get_and_increment() for index in partial_indices}
          get_partial_index = lambda index: index[:self.disjunctive_dim] + index[self.disjunctive_dim + 1:]
          facts = [((tensor[index], disj_map[get_partial_index(index)]), self[index]) for index in filtered_indices]
        else:
          disj_id = mutual_exclusion_counter.get_and_increment()
          facts = [((tensor[index], disj_id), self[index]) for index in filtered_indices]
      else:
        facts = [((tensor[index], None), self[index]) for index in filtered_indices]
    else:
      facts = [(tensor[index], self[index]) for index in filtered_indices]

    # Return facts
    return facts

  def _initialize_list_mapping(self, mapping):
    """
    Initialize an input mapping using a list

    The list can be nested, in which case it will be treated as a multi-dimensional mapping
    """

    # Need to make sure that a mapping is not empty
    if len(mapping) == 0:
      raise Exception("Invalid input mapping: a mapping list cannot be empty")

    # To process list of elements
    curr_elements = [e for e in mapping]
    shape = [len(curr_elements)]
    is_singleton = None
    while True:
      first_element = curr_elements[0]
      first_element_ty = type(first_element)
      if self._is_primitive_or_tuple(first_element):
        tuple_size = None if self._is_primitive(first_element) else len(first_element)

        # The first element is primitive or tuple; that means all elements need to be value or tuple
        for (i, element) in enumerate(curr_elements):
          if not self._is_primitive_or_tuple(element):
            raise Exception(f"Invalid input mapping: expected terminal value/tuple at dimension {len(shape) + 1}, found {type(element)}")

          # Check the consistency of the tuple
          if tuple_size == None:
            if not self._is_primitive(element):
              raise Exception(f"Invalid input mapping: expected singleton value, found tuple")
            curr_elements[i] = (element,)
          else:
            if len(element) != tuple_size:
              raise Exception(f"Invalid input mapping: expected tuple size {tuple_size}, found {len(element)}")

        # If all checks out, then set is_singleton value
        is_singleton = tuple_size is None

        # Hit a good base case
        break

      elif first_element_ty == list or first_element_ty == range:
        # The first element is a list; first, the length of the list cannot be zero
        next_size = len(first_element)
        if next_size == 0:
          raise Exception(f"Invalid input mapping: having potential empty dimension {len(shape) + 1}")

        # that means all elements need to be list of the same size
        for element in curr_elements:
          if type(element) != list:
            if type(element) == range:
              element = list(element)
            else:
              raise Exception(f"Invalid input mapping: expected a list or range, found {type(element)} at dimension {len(shape) + 1}")
          if len(element) != next_size:
            raise Exception(f"Invalid input mapping: shape mismatch at dimension {len(shape) + 1}, expected {next_size}, found {len(element)}")

        # Hit a good recursive case
        shape.append(next_size)
        curr_elements = [grand_child for child in curr_elements for grand_child in child]

      else:
        # Not okay
        raise Exception(f"Invalid input mapping: encountered element {type(first_element)}")

    # Upon success, set the basic information
    self._kind = "list"
    self._shape = tuple(shape)
    self._is_singleton = is_singleton
    self._source = mapping
    self._mapping = curr_elements

  def _initialize_dict_mapping(self, mapping: Dict[int, List]):
    """Initialize the input mapping with a dictionary"""

    # Mapping cannot be empty
    if len(mapping) == 0:
      raise Exception("Invalid input mapping: cannot be empty")

    # Check if the keys in the mapping are proper dimension integers
    for dim_num in mapping.keys():
      if type(dim_num) != int:
        raise Exception(f"Invalid input mapping: invalid dimension {dim_num}")
      if dim_num < 0:
        raise Exception("Invalid input mapping: cannot have negative dimension number")

    # Check the maximum dimensions and all dimensions are presented
    max_dim_num = max(mapping.keys())
    if len(mapping) != max_dim_num + 1:
      for i in range(max_dim_num + 1):
        if i not in mapping:
          raise Exception(f"Invalid input mapping: missing dimension {i}")

    # Get a shape
    dimension = max_dim_num + 1

    # Check that the values of the dictionary are all lists; create the shape list
    processed_mapping = []
    shape = []
    for i in range(dimension):
      to_check = mapping[i]
      if type(to_check) != list:
        if type(to_check) == range:
          to_check = list(to_check)
        else:
          raise Exception("Invalid input mapping: value of dictionary must be a list or a range")
      if len(to_check) == 0:
        raise Exception(f"Invalid input mapping: empty dimension {i}")
      for element in to_check:
        if not self._is_primitive(element):
          raise Exception("Invalid input mapping: element of dictionary value must be a primitive")
      shape.append(len(to_check))
      processed_mapping.append(to_check)

    # Success!
    self._kind = "dict"
    self._shape = tuple(shape)
    self._is_singleton = False
    self._source = mapping
    self._mapping = processed_mapping

  def _initialize_tuple_mapping(self, mapping):
    """Initialize the input mapping with a tuple."""

    # Check if all the elements inside of the tuple is value
    for element in mapping:
      if not self._is_primitive(element):
        raise Exception("Invalid input mapping: elements of tuple must be a primitive")

    # Success!
    self._kind = "tuple"
    self._shape = tuple()
    self._is_singleton = False
    self._source = mapping
    self._mapping = [mapping]

  def _initialize_value_mapping(self, mapping):
    """Initialize the input mapping with a value"""
    self._kind = "value"
    self._shape = tuple()
    self._is_singleton = True
    self._source = mapping
    self._mapping = [(mapping,)]

  def _is_primitive_or_tuple(self, e) -> bool:
    ty = type(e)
    if ty == tuple:
      for ce in e:
        if not self._is_primitive(ce):
          return False
      return True
    else:
      return self._is_primitive(e)

  def _mult_dim_index_to_index(self, mult_dim_index):
    summed_index = 0
    for i in range(len(self._shape)):
      summed_index += mult_dim_index[i] * reduce(lambda acc, i: acc * i, self._shape[i + 1:], 1)
    return summed_index

  def _index_to_mult_dim_index(self, index):
    acc_index = index
    mult_dim_index = []
    for i in range(len(self._shape), 0, -1):
      mult_dim_index.append(acc_index % self._shape[i - 1])
      acc_index = acc_index // self._shape[i - 1]
    return tuple(reversed(mult_dim_index))

  def _convert_topk_sampled_indices(self, sampled_indices, sample_dim):
    for partial_index in itertools.product(*[range(d) for (i, d) in enumerate(self._shape) if i != sample_dim]):
      before, after = partial_index[:sample_dim], partial_index[sample_dim:]
      indexing_slice = before + (slice(None),) + after
      for sampled_k_index in sampled_indices[indexing_slice]:
        yield before + (int(sampled_k_index),) + after

  def _convert_categorical_sampled_indices(self, sampled_indices, sample_dim):
    for partial_index in itertools.product(*[range(d) for (i, d) in enumerate(self._shape) if i != sample_dim]):
      before, after = partial_index[:sample_dim], partial_index[sample_dim:]
      indexing_slice = before + after
      for sampled_k_index in sampled_indices[indexing_slice]:
        yield before + (int(sampled_k_index),) + after

  def _is_primitive(self, e) -> bool:
    ty = type(e)
    return ty == int or ty == float or ty == str or ty == bool

  def _get_kind(self) -> str:
    return self._kind

  def _set_kind(self):
    raise Exception("Cannot set kind of an input mapping")

  kind = property(_get_kind, _set_kind)

  def _get_shape(self) -> Tuple[int]:
    return self._shape

  def _set_shape(self):
    raise Exception("Cannot set shape of an input mapping")

  shape = property(_get_shape, _set_shape)

  def _get_dimension(self) -> int:
    return len(self._shape)

  def _set_dimension(self):
    raise Exception("Cannot set dimension of an input mapping")

  dimension = property(_get_dimension, _set_dimension)

  def _get_is_singleton(self) -> int:
    return self._is_singleton

  def _set_is_singleton(self):
    raise Exception("Cannot set is_singleton of an input mapping")

  is_singleton = property(_get_is_singleton, _set_is_singleton)


class InclusiveSet:
  def __init__(self, init_include_all=True):
    self.include_all = init_include_all
    self.included_set = set()

  def add(self, index: Tuple):
    self.include_all = False
    self.included_set.add(index)

  def contains(self, index: Tuple):
    return self.include_all or (index in self.included_set)


class ExclusiveSet:
  def __init__(self):
    self.excluded_set = set()

  def exclude(self, index: Tuple):
    self.excluded_set.add(index)

  def contains(self, index: Tuple) -> bool:
    return index not in self.excluded_set
