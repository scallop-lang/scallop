from __future__ import annotations
from typing import List, Union, Tuple, Optional, Any, Dict, Callable
from copy import deepcopy

# Try import torch; if not there, delegate to something else
from .torch_importer import *

# Import internals
from .scallopy import InternalScallopContext
from .collection import ScallopCollection
from .provenance import ScallopProvenance, DiffAddMultProb2Semiring, DiffNandMultProb2Semiring, DiffMaxMultProb2Semiring
from .io import CSVFileOptions
from .utils import _mapping_tuple
from .history import HistoryAction, record_history
from .sample_type import SAMPLE_TYPE_TOP_K

# Main context
class ScallopContext(Context):
  """
  A Scallop execution context that fosters all compilation and execution.

  Usage:

  ``` python
  ctx = ScallopContext()
  ```

  :param provenance: The type of provenance used during execution.
  Default to "unit", and can be any value from the following
  - `"unit"`, no provenance information associated
  - `"proofs"`, collect proofs
  - `"minmaxprob"`, min-max probability
  - `"addmultprob"`, add-mult probability
  - `"topkproofs"`, top-k proofs. It is possible to supply a `k` value for this
    provenance
  - `"diffminmaxprob"`, differentiable min-max probability
  - `"diffandmultprob"`, differentiable add-mult probability
  - `"diffandmultprob2"`, differentiable add-mult probability, implemented in Python
  - `"diffnandmultprob2"`, differentiable nand-mult probability, implemented in Python
  - `"diffmaxmultprob2"`, differentiable max-mult probability, implemented in Python
  - `"diffsamplekproofs"`, differentiable unbiased sampling of k proofs. It is
    possible to supply a `k` value for this provenance
  - `"difftopkproofs"`, differentiable top-k proofs. It is possible to supply a
    `k` value for this provenance
  - `"difftopbottomkclauses"`, differentiable top/bottom-k clauses.
    This provenance supports full negation and aggregation. It is possible to supply
    a `k` value for this provenance

  :param k:
  :param train_k:
  :param test_k:
  """
  def __init__(
    self,
    provenance: str = "unit",
    custom_provenance: Optional[ScallopProvenance] = None,
    k: int = 3,
    train_k: Optional[int] = None,
    test_k: Optional[int] = None,
    fork_from: Optional[ScallopContext] = None
  ):
    super(ScallopContext, self).__init__()

    # Check if we are creating a forked context or not
    if fork_from is None:
      # Validify provenance; differentiable needs PyTorch
      if "diff" in provenance and not has_pytorch:
        raise Exception("Attempting to use differentiable provenance but with no PyTorch imported")

      # Prepare Python based provenance
      if provenance == "diffaddmultprob2":
        provenance = "custom"
        custom_provenance = DiffAddMultProb2Semiring()
      elif provenance == "diffnandmultprob2":
        provenance = "custom"
        custom_provenance = DiffNandMultProb2Semiring()
      elif provenance == "diffmaxmultprob2":
        provenance = "custom"
        custom_provenance = DiffMaxMultProb2Semiring()
      else: pass

      # Setup
      self.provenance = provenance
      self._custom_provenance = custom_provenance
      self._input_mappings = {}
      self._input_retain_topk = {}
      self._input_non_probabilistic = {}
      self._input_is_singleton = {}
      self._sample_facts = {}
      self._k = k
      self._train_k = train_k
      self._test_k = test_k
      self._history_actions: List[HistoryAction] = []
      self._internal = InternalScallopContext(provenance=provenance, custom_provenance=custom_provenance, k=k)
    else:
      # Fork from an existing context
      self.provenance = deepcopy(fork_from.provenance)
      self._custom_provenance = deepcopy(fork_from._custom_provenance)
      self._input_mappings = deepcopy(fork_from._input_mappings)
      self._input_retain_topk = deepcopy(fork_from._input_retain_topk)
      self._input_non_probabilistic = deepcopy(fork_from._input_non_probabilistic)
      self._input_is_singleton = deepcopy(fork_from._input_is_singleton)
      self._sample_facts = deepcopy(fork_from._sample_facts)
      self._k = deepcopy(fork_from._k)
      self._train_k = deepcopy(fork_from._train_k)
      self._test_k = deepcopy(fork_from._test_k)
      self._history_actions = deepcopy(fork_from._history_actions)
      self._internal = fork_from._internal.clone()

  def __getstate__(self):
    """
    Serialize into pickle state
    """
    state = self.__dict__.copy()
    del state["_internal"]
    return state

  def __setstate__(self, state):
    """
    Deserialize from pickle state
    """
    # Serializable part
    self.__dict__.update(state)

    # Initialize empty history actions; this will be populated by later restoration
    self._history_actions: List[HistoryAction] = []

    # Internal scallop context
    self._internal = InternalScallopContext(provenance=self.provenance, custom_provenance=self._custom_provenance, k=self._k)

    # Restore from history actions
    for history_action in state["_history_actions"]:
      function = getattr(self, history_action.func_name)
      function(*history_action.pos_args, **history_action.kw_args)

  def clone(self) -> ScallopContext:
    """
    Clone the current context. This is useful for incremental execution:

    ``` python
    ctx2 = ctx.clone()
    ctx2.add_rule("...")
    ctx2.run()

    ctx3 = ctx.clone()
    ctx3.add_rule("...")
    ctx3.run()
    ```

    In this example, `ctx2` and `ctx3` will be executed independently,
    but both could inherit the computation already done on `ctx`.
    """
    return ScallopContext(fork_from=self)

  def run(self, iter_limit: Optional[int] = None):
    """
    Execute the code under the current context. This operation is incremental
    and might use as many as previous information as possible.

    ``` python
    ctx.run()
    ```
    """
    self._internal.run(iter_limit)

  def run_batch(
    self,
    inputs: Dict[str, List[Union[Tuple[List[Tuple], Optional[List[List[int]]]], List[Tuple]]]],
    output_relation: Union[str, List[str]],
    parallel: bool = False,
    iter_limit: Optional[int] = None,
  ):
    """
    Run the program in batch mode
    """
    batch_size = len(list(inputs.values())[0])
    output_relations = output_relation if type(output_relation) == list else [output_relation] * batch_size
    inputs = {r: [i if type(i) == tuple else (i, None) for i in b] for (r, b) in inputs.items()}
    internal_collections = self._internal.run_batch(iter_limit, output_relations, inputs, parallel)
    return [ScallopCollection(self.provenance, coll) for coll in internal_collections]

  @record_history
  def import_file(self, file_name: str):
    """
    Import a file given the file name
    """
    self._internal.import_file(file_name)

  @record_history
  def add_program(self, program: str):
    """
    Add a full scallop program string
    """
    self._internal.add_program(program)

  def forward_function(
    self,
    output: Optional[str] = None,
    output_mapping: Optional[Union[List[Tuple], Tuple]] = None,
    output_mappings: Optional[Dict[str, List[Tuple]]] = None,
    iter_limit: Optional[int] = None,
    dispatch: Optional[str] = "parallel",
    debug_provenance: bool = False,
    retain_graph: bool = False,
    jit: bool = False,
    jit_name: str = "",
    recompile: bool = False,
  ) -> Callable:
    """
    Generate a forward function for PyTorch module.

    Example:

    ``` python
    eval = ctx.forward_function("target")
    result = eval(input_1=tensor_1, input_2=tensor_2)
    ```

    :param output: the output relation name
    :param output_mapping: the output mapping for vectorization
    :param iter_limit: the amount of maximum iteration.
    If not specified (i.e. None), will run until fixpoint
    """
    # Import ScallopForward to avoid circular dependency
    from .forward import InternalScallopForwardFunction

    # Check PyTorch support
    if not has_pytorch:
      raise Exception("`forward_function` cannot be called when there is no PyTorch")

    # Needs to be a differentiable context
    if ("diff" in self.provenance or self.provenance == "custom") and not "debug" in self.provenance: pass
    else: raise Exception("`forward_function` can only be called on context with differentiable provenance")

    # Forward function
    return InternalScallopForwardFunction(self, output, output_mapping, output_mappings, iter_limit, dispatch, debug_provenance, retain_graph, jit, jit_name, recompile)

  def _refresh_training_eval_state(self):
    if self._train_k is not None or self._test_k is not None:
      if self.training: # Coming from nn.Module and `torch.train()` or `torch.eval()`
        if self._train_k is not None:
          self._internal.set_k(self._train_k)
        else:
          self._internal.set_k(self._k)
      else:
        if self._test_k is not None:
          self._internal.set_k(self._test_k)
        else:
          self._internal.set_k(self._k)

  @record_history
  def add_relation(
    self,
    relation_name: str,
    relation_types: Union[Tuple, type, str],
    input_mapping: Optional[Union[List[Tuple], Tuple]] = None,
    retain_topk: Optional[int] = None,
    non_probabilistic: bool = False,
    load_csv: Optional[Union[CSVFileOptions, str]] = None,
  ):
    """
    Add a relation to the context, where a relation is defined by its name
    and the tuple types.
    Idiomatic python types such as `str`, `int`, `bool` are supported, while
    internally being transformed into corresponding scallop types.
    (e.g. `int` -> `i32`).

    ``` python
    ctx.add_relation("edge", (int, int))
    ctx.add_relation("path", (int, int))
    ctx.add_relation("digit", int)
    ```

    Note that we usually accept a tuple. When a single element, such as `int`
    is provided, it will be internally transformed into a one-tuple.

    In addition to idiomatic python types, you can use the types provided in
    `types` module to gain access to native Scallop types:

    ``` python
    from scallopy import types
    ctx.add_relation("digit", (types.i8,))
    ```

    Of course, directly feeding type as string is ok too:

    ``` python
    ctx.add_relation("digit", "i8")
    ```

    You can load csv using this function

    ``` python
    ctx.add_relation(
      "edge", (int, int),
      load_csv = "FILE.csv",
    )
    ```

    If you wish to specify properties of CSV such as deliminator, you can
    use the `CSVFile` class

    ``` python
    edge_csv_file = CSVFile("FILE.csv", deliminator="\t")
    ctx.add_relation("edge", (int, int), load_csv=edge_csv_file)
    ```
    """

    # Helper function
    def _type_to_scallop_type_str(ty):
      if ty == str:
        return "String"
      elif ty == int:
        return "i32"
      elif ty == bool:
        return "bool"
      elif ty == float:
        return "f32"
      elif type(ty) == str:
        return ty
      else:
        raise Exception(f"Unknown type `{ty}`")

    # Make sure that relation types is a tuple
    is_singleton_tuple = False
    if type(relation_types) == tuple:
      relation_types_tuple = relation_types
    elif type(relation_types) == type or type(relation_types) == str:
      is_singleton_tuple = True
      relation_types_tuple = (relation_types,)
    else:
      raise Exception(f"Unknown relation types `{relation_types}`")

    # Create the decl str
    types_str = ", ".join([_type_to_scallop_type_str(ty) for ty in relation_types_tuple])
    relation_decl_str = f"{relation_name}({types_str})"

    # Invoke internal's add relation
    inserted_relation_name = self._internal.add_relation(relation_decl_str, load_csv=load_csv)

    # Sanity check
    assert relation_name == inserted_relation_name

    # Store the input mapping
    if input_mapping is not None:
      self.set_input_mapping(relation_name, input_mapping)

    # Store the retain topk property
    if retain_topk is not None:
      self._input_retain_topk[relation_name] = retain_topk

    # Store the non-probabilistic property
    if non_probabilistic:
      self._input_non_probabilistic[relation_name] = True

    # Store the input is singleton property
    if is_singleton_tuple:
      self._input_is_singleton[relation_name] = True

  @record_history
  def add_facts(
    self,
    relation: str,
    elems: List[Tuple],
    disjunctions: Optional[List[List[int]]] = None,
  ):
    """
    Add facts to the relation under the context. The simple usage is as
    follows:

    ``` python
    ctx.add_facts("edge", [(0, 1), (1, 2)])
    ```

    Note that when adding relation, the relation name must be previously
    defined in the context using `add_relation` or by inferred from other
    rules. The function may throw error if the type does not match.

    When the context is associated with a non-unit provenance context,
    say "minmaxprob", one would need to provide the tag associated with
    each tuple. For "minmaxprob", as an example, one would need to invoke
    the function like this:

    ``` python
    ctx.add_facts("digit", [
      (0.90, (0, 1)),
      (0.01, (0, 2)),
      (0.03, (0, 3)),
    ])
    ```

    Note that there is a probability in the beginning and each tuple
    is now nested.

    :param relation: the name of the relation
    :param elems: the tuple elements
    :param disjunctions: the disjunctions
    """

    # First normalize the probability format
    if relation in self._input_non_probabilistic and self._input_non_probabilistic[relation] and self.requires_tag():
      elems = [(None, t) for t in elems]

    # Then normalize the singleton tuple format
    if relation in self._input_is_singleton and self._input_is_singleton[relation]:
      if self.requires_tag():
        elems = [(tag, (tup,)) if type(tup) != tuple else (tag, tup) for (tag, tup) in elems]
      else:
        elems = [(tup,) if type(tup) != tuple else tup for tup in elems]

    # Add the facts
    self._internal.add_facts(relation, elems, disjunctions)

  @record_history
  def add_rule(self, rule: str, tag: Optional[Any] = None, demand: Optional[str] = None):
    """
    Add rule to the context. The rule will be compiled and compilation
    error may be raised.

    ``` python
    ctx.add_rule("path(a, c) = edge(a, b), path(b, c)")
    ```

    In case non-unit provenance is used, a rule can be associated with
    tag. For example, in a probabilistic provenance context, one can
    associate probability with the rule.

    ``` python
    ctx.add_rule("born_in(a, \"china\") :- speaks(a, \"chinese\")", tag = 0.8)
    ```

    :param rule: a rule in scallop syntax
    :param tag: the tag associated with the rule
    """
    self._internal.add_rule(rule, tag=tag, demand=demand)

  @record_history
  def compile(self):
    self._internal.compile()

  def dump_front_ir(self):
    """
    Dump the Scallop front internal representation of the program compiled
    inside this context.
    """
    self._internal.dump_front_ir()

  def relation(self, relation: str, debug: bool = False) -> ScallopCollection:
    """
    Inspect the (computed) relation in the context. Will return
    a `ScallopCollection` which is iterable.

    ``` python
    for tuple in ctx.relation("edge"):
      print(tuple)
    ```

    :param relation: the name of the relation
    """
    int_col = self._internal.relation(relation) if not debug else self._internal.relation_with_debug_tag(relation)
    return ScallopCollection(self.provenance, int_col)

  def has_relation(self, relation: str) -> bool:
    """
    Check if the compiled program contains a given relation.
    """
    return self._internal.has_relation(relation)

  def relation_is_computed(self, relation: str) -> bool:
    """
    Check if the relation is computed.
    """
    return self._internal.relation_is_computed(relation)

  def num_relations(self, include_hidden: bool = False) -> int:
    """
    Get the number of relations in the context.
    """
    return self._internal.num_relations(include_hidden=include_hidden)

  def relations(self, include_hidden: bool = False) -> List[str]:
    """
    Get a list of user defined relations in the context.
    If `include_hidden` is specified to be `True`, will also return the auxilliary
    relations (such as sub-formula, permutation, etc) in the list.
    """
    return self._internal.relations(include_hidden=include_hidden)

  def set_non_probabilistic(self, relation: Union[str, List[str]], non_probabilistic: bool = True):
    """
    Set the relation to be non-probabilistic
    """
    if type(relation) == list:
      for r in relation:
        self._set_relation_non_probabilistic(r, non_probabilistic)
    else:
      self._set_relation_non_probabilistic(relation, non_probabilistic)

  def _set_relation_non_probabilistic(self, relation: str, non_probabilistic: bool):
    if self.has_relation(relation):
      self._input_non_probabilistic[relation] = non_probabilistic
    else:
      raise Exception(f"Unknown relation {relation}")

  def set_input_mapping(self, relation: str, input_mapping: Union[List[Tuple], Tuple]):
    """
    Set the input mapping for the given relation
    """
    if type(input_mapping) == list:
      (preproc_input_mapping, is_singleton) = ([_mapping_tuple(t) for t in input_mapping], False)
    elif type(input_mapping) == tuple:
      (preproc_input_mapping, is_singleton) = ([_mapping_tuple(input_mapping)], True)
    else:
      raise Exception(f"Unknown input mapping type `{type(input_mapping)}`. Must be a list or tuple")

    # Check if the tuples in the input mapping matches
    for tuple in preproc_input_mapping:
      if not self._internal.check_tuple(relation, tuple):
        raise Exception(f"The tuple {tuple} in the input mapping does not match the type of the relation `{relation}`")

    # If there is no problem, set the input_mapping property
    self._input_mappings[relation] = (preproc_input_mapping, is_singleton)

  def set_sample_topk_facts(self, relation: str, amount: int):
    self._input_retain_topk[relation] = amount

  def set_sample_facts(self, relation: str, amount: int, sample_type: str = SAMPLE_TYPE_TOP_K):
    """
    When forward function is called, sample from the given facts by the amount

    :param relation: the relation within which the facts need to be sampled
    :param sample_type: can be chosen from "categorical" or ""
    """
    self._sample_facts[relation] = (sample_type, amount)

  def requires_tag(self) -> bool:
    """
    Returns whether the context requires facts to be associated with tags
    """
    if self.provenance == "unit" or self.provenance == "proofs":
      return False
    else:
      return True
