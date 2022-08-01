from typing import Dict, Union, List, Optional, Tuple, Any, Callable

from .torch_importer import *
from .context import ScallopContext
from .utils import _mapping_tuple

class ScallopForward(torch.nn.Module):
  """
  A Scallop PyTorch forward function.

  :param dispatch str, can be chosen from `"single"`, `"batch"`, or `"parallel"`
  """
  def __init__(
    self,
    ctx: ScallopContext,
    output: Optional[str] = None,
    output_mapping: Optional[Union[List[Tuple], Tuple]] = None,
    iter_limit: Optional[int] = None,
    dispatch: str = "parallel",
    debug_provenance: bool = False,
    retain_graph: bool = False,
  ):
    super(ScallopForward, self).__init__()

    # Parameters
    self.ctx = ctx
    self.output = output
    self.iter_limit = iter_limit
    self.single_element = False
    self.dispatch = dispatch
    self.debug_provenance = debug_provenance
    self.retain_graph = retain_graph

    # Default tensor apply function is an identity function; later it might be
    # a `to("cpu")` or `to("gpu")` function.
    self._torch_tensor_apply = lambda x: x

    # Process the mapping list
    if type(output_mapping) == list:
      self.output_mapping = [_mapping_tuple(t) for t in output_mapping]
    elif type(output_mapping) == tuple:
      self.output_mapping = [_mapping_tuple(output_mapping)]
      self.single_element = True
    elif output_mapping is None:
      self.output_mapping = None
    else:
      raise Exception(f"Unknown output mapping type `{type(output_mapping)}`")

  def __getstate__(self):
    """
    Serialize into pickle state
    """
    state = self.__dict__.copy()
    del state["_torch_tensor_apply"]
    return state

  def __setstate__(self, state):
    """
    Deserialize from pickle state
    """
    # Serializable part
    self.__dict__.update(state)
    self._torch_tensor_apply = lambda x: x

  def _apply(self, f):
    """
    Overriding `nn.Module`'s _apply function to accept additional tensor apply
    function `f`. Usually this `f` function is a `.to("cpu")` or `.to("gpu")`
    function. For Scallop to support GPU, we will use this function `f` to
    apply to any tensor that we generate in the computation process.
    """
    self._torch_tensor_apply = f

  def __call__(
    self,
    disjunctions: Optional[Dict[str, List[List[List[int]]]]] = None,
    output_relations: Optional[List[str]] = None,
    **input_facts: Dict[str, Union[Tensor, List]],
  ) -> Union[Tensor, Tuple[List[Tuple], Tensor]]:
    """
    Invoke the forward function with the given facts

    The facts and disjunctions need to be batched
    """

    self.ctx._refresh_training_eval_state() # Set train/eval
    self.ctx._internal.compile() # Compile into back IR

    # First make sure that all facts share the same batch size
    batch_size = self._compute_and_check_batch_size(input_facts)

    # Process the input into a unified form
    all_inputs = self._process_all_input_facts(batch_size, input_facts, disjunctions)

    # Process the output into a list of output relations
    output_relations = output_relations if output_relations is not None else [self.output] * batch_size
    if len(output_relations) != batch_size:
      raise Exception(f"Number of output relations ({len(output_relations)}) does not match the batch size ({batch_size})")

    # Check task dispatcher
    if self.dispatch == "single":
      # Execute scallop program for each task from python
      input_tags = []
      output_results = []
      for task_id in range(batch_size):
        (task_input_tags, task_output_results) = self._run_single(task_id, all_inputs, output_relations[task_id])
        input_tags.append(task_input_tags)
        output_results.append(task_output_results)
    elif self.dispatch == "batch":
      # Directly dispatch all the inputs to rust
      (input_tags, output_results) = self._run_batch(batch_size, all_inputs, output_relations, parallel=False)
    elif self.dispatch == "parallel":
      # Dispatch all the inputs to rust and call rayon as parallelism backend
      (input_tags, output_results) = self._run_batch(batch_size, all_inputs, output_relations, parallel=True)
    else:
      raise Exception(f"Unknown dispatch type `{self.dispatch}`")

    # Process the output
    return self._process_output(batch_size, input_tags, output_results)

  def _compute_and_check_batch_size(self, inputs: Dict[str, Union[Tensor, List]]) -> int:
    """
    Given the inputs, check if the batch size is consistent over all relations.
    If so, return the batch size.
    Otherwise, when there is no input or any batch size is inconsistent, throw exception.
    """
    batch_size = None
    for (rela, rela_facts) in inputs.items():
      if batch_size is None:
        batch_size = len(rela_facts)
      elif batch_size != len(rela_facts):
        raise Exception(f"Inconsistency in batch size: expected {batch_size}, found {len(rela_facts)} for relation `{rela}`")
    if batch_size is None:
      raise Exception("There is no input to the forward function")
    else:
      return batch_size

  def _process_all_input_facts(self, batch_size, all_input_facts, all_disjunctions):
    """
    Given all the input facts where facts may be lists or tensors,
    process them into a unified form.
    """
    processed_inputs = {}
    for (rela, rela_facts) in all_input_facts.items():
      processed_inputs[rela] = []
      for task_id in range(batch_size):
        ds = all_disjunctions[rela][task_id] if all_disjunctions is not None and rela in all_disjunctions else None
        (facts, ds) = self._process_input_facts(rela, rela_facts[task_id], ds)
        processed_inputs[rela].append((facts, ds))
    return processed_inputs

  def _process_input_facts(self, rela, rela_facts, disjunctions) -> Tuple[List[Tuple], Optional[List[List[int]]]]:
    """
    Given input facts of one single relation (and its disjunctions), process it into
    a unified form of input facts along with its disjunctions.

    Note that the disjunction IDs will be normalized as well.
    """
    if rela in self.ctx._input_non_probabilistic and self.ctx._input_non_probabilistic[rela]:
      # Add non-probabilistic facts; there will be no disjunctions
      return ([(None, f) for f in rela_facts], None)
    else:
      # Process the facts
      ty = type(rela_facts) # The type of relation facts
      index_mapping = None # The index mapping of given facts and preprocessed facts if there is removal of facts
      if ty == list:
        facts = rela_facts
        if rela in self.ctx._input_retain_topk:
          k = min(self.ctx._input_retain_topk[rela], len(facts))
          if disjunctions is not None:
            indexed_facts = sorted(enumerate(facts), key=lambda x: x[1][0].item(), reverse=True)[:k]
            index_mapping = {j: i for (i, (j, _)) in enumerate(indexed_facts)}
            facts = [f for (_, f) in indexed_facts]
          else:
            facts = sorted(facts, key=lambda x: x[0].item(), reverse=True)[:k]
      elif ty == Tensor:
        if rela not in self.ctx._input_mappings:
          raise Exception(f"scallopy.forward receives vectorized Tensor input. However there is no `input_mapping` provided for relation `{rela}`")
        probs = rela_facts
        single_element = self.ctx._input_mappings[rela][1]
        if single_element:
          fact = self.ctx._input_mappings[rela][0][0]
          facts = [(probs, fact)]
        else:
          if rela in self.ctx._input_retain_topk:
            k = min(self.ctx._input_retain_topk[rela], len(probs))
            (top_probs, top_prob_ids) = torch.topk(probs, k)
            facts = [(p, self.ctx._input_mappings[rela][0][i]) for (p, i) in zip(top_probs, top_prob_ids)]
            if disjunctions is not None:
              index_mapping = {j: i for (i, j) in enumerate(top_prob_ids)}
          else:
            facts = list(zip(probs, self.ctx._input_mappings[rela][0]))
      else:
        raise Exception(f"Unknown input facts type. Expected Tensor or List, found {ty}")

      # Remap disjunction
      remapped_disjs = [[index_mapping[i] for i in d if i in index_mapping] for d in disjunctions] if index_mapping is not None else disjunctions

      # Add the facts
      return (facts, remapped_disjs)

  def _run_single(self, task_id, all_inputs, output_relation):
    """
    Run a single task (identified by `task_id`)

    Internally we will pick out the inputs corresponding to this task only.
    The task is manually dispatched using internal Scallopy python API.
    """

    # Clone a context for scallop execution
    temp_ctx = self.ctx.clone()

    # Add the facts into the context
    for (rela, rela_inputs) in all_inputs.items():
      if not self.ctx.has_relation(rela):
        raise Exception(f"Unknown relation `{rela}`")
      (facts, disjunctions) = rela_inputs[task_id]
      temp_ctx.add_facts(rela, facts, disjunctions=disjunctions)

    # Execute the context
    if self.debug_provenance:
      temp_ctx._internal.run_with_debug_tag(iter_limit=self.iter_limit)
    else:
      temp_ctx.run(iter_limit=self.iter_limit)

    # Get input tags
    input_tags = temp_ctx._internal.input_tags()

    # Get the internal collection for the target output
    if self.debug_provenance:
      internal_collection = temp_ctx._internal.relation_with_debug_tag(output_relation)
    else:
      internal_collection = temp_ctx._internal.relation(output_relation)

    # Process the collection to get the output results
    output_results = self._process_single_output(internal_collection)

    # Return
    return (input_tags, output_results)

  def _run_batch(self, batch_size, all_inputs, output_relations, parallel: bool):
    """
    Run a batch of tasks
    """
    result = self.ctx._internal.run_batch(self.iter_limit, output_relations, all_inputs, parallel)
    input_tags, output_results = [], []
    for task_id in range(batch_size):
      input_tags.append(result[task_id].input_tags())
      output_results.append(self._process_single_output(result[task_id]))
    return (input_tags, output_results)

  def _process_single_output(self, internal_collection):
    internal_result_dict = { tup: tag for (tag, tup) in internal_collection }
    if self.output_mapping is not None:
      return [internal_result_dict[t] if t in internal_result_dict else None for t in self.output_mapping]
    else:
      return internal_result_dict

  def _process_output(self, batch_size, input_tags, output_results):
    # If there is no given output mapping, try
    if self.output_mapping is not None:
      # Integrate the outputs
      v = self._batched_prob(output_results)
      if self._has_output_hook() and v.requires_grad:
        v.register_hook(self._batched_output_hook(input_tags, output_results))
      v = v.view(-1) if self.single_element else v

      # Return the output
      return v
    else:
      # Collect all possible outputs, make them into a list
      possible_outputs = set()
      for internal_output_result in output_results:
        for (o, _) in internal_output_result.items():
          possible_outputs.add(o)
      post_output_mapping = list(possible_outputs)

      # Make the result into a batch
      post_output_results = []
      for internal_result_dict in output_results:
        post_output_results.append([internal_result_dict[t] if t in internal_result_dict else None for t in post_output_mapping])

      # Check if there is no output result
      if len(possible_outputs) == 0:
        # Return empty tensor
        return ([], torch.zeros(batch_size, 0, requires_grad=self.training))
      else:
        # Integrate the outputs based on all the output results
        v = self._batched_prob(post_output_results)
        if self._has_output_hook() and v.requires_grad:
          v.register_hook(self._batched_output_hook(input_tags, post_output_results))
        v = v.view(-1) if self.single_element else v

        # Return
        return (post_output_mapping, v)

  def _batched_prob(
    self,
    tasks: List[List[Any]],
  ) -> Tensor:
    # Provenance diffminmaxprob
    if self.ctx.provenance == "diffminmaxprob":
      tensor_results = []
      for task_results in tasks:
        task_tensor_results = []
        for task_tup_result in task_results:
          if task_tup_result is not None:
            (s, p) = task_tup_result
            if s == 1:
              task_tensor_results.append(p)
            elif s == 0:
              task_tensor_results.append(self._torch_tensor_apply(torch.tensor(p, requires_grad=True)))
            else:
              task_tensor_results.append(1 - p)
          else:
            task_tensor_results.append(self._torch_tensor_apply(torch.tensor(0.0, requires_grad=True)))
        tensor_results.append(torch.stack(task_tensor_results))
      return torch.stack(tensor_results)

    # Provenance diff addmultprob / proofs
    # -- the provenances that returns a full derivatives array associated with the probability
    elif self.ctx.provenance == "diffaddmultprob" or \
         self.ctx.provenance == "difftopkproofs" or \
         self.ctx.provenance == "diffsamplekproofs" or \
         self.ctx.provenance == "difftopbottomkclauses":
      tensor_results = []
      for task_results in tasks:
        task_tensor_results = []
        for task_tup_result in task_results:
          if task_tup_result is not None:
            (p, _) = task_tup_result
            task_tensor_results.append(torch.tensor(p, requires_grad=True))
          else:
            task_tensor_results.append(torch.tensor(0.0, requires_grad=True))
        tensor_results.append(torch.stack(task_tensor_results))
      return self._torch_tensor_apply(torch.stack(tensor_results))

    # If we have a custom provenance, use its collate function
    elif self.ctx.provenance == "custom":
      return self.ctx._custom_provenance.collate(tasks)

    # Non-differentiable provenance
    else:
      raise Exception("[Internal Error] Should not happen")

  def _has_output_hook(self):
    if self.ctx.provenance == "diffaddmultprob": return True
    elif self.ctx.provenance == "difftopkproofs": return True
    elif self.ctx.provenance == "diffsamplekproofs": return True
    elif self.ctx.provenance == "difftopbottomkclauses": return True
    else: return False

  def _batched_output_hook(
    self,
    input_tags: Optional[List[List[Any]]],
    tasks: List[List[Any]],
  ) -> Callable:
    if self.ctx.provenance == "diffminmaxprob":
      raise Exception("`diffminmaxprob` does not implement batched output hook")
    elif self.ctx.provenance == "diffaddmultprob" or \
         self.ctx.provenance == "difftopkproofs" or \
         self.ctx.provenance == "diffsamplekproofs" or \
         self.ctx.provenance == "difftopbottomkclauses":
      return self._diff_proofs_batched_output_hook(input_tags, tasks)
    else:
      raise Exception("[Internal Error] Should not happen")

  def _diff_proofs_batched_output_hook(
    self,
    input_tags: List[List[Any]],
    output_batch: List[List[Any]],
  ) -> Callable:
    # Prepare the dimensions
    batch_size = len(output_batch)
    num_inputs = max(len(ts) for ts in input_tags)
    num_outputs = len(output_batch[0])

    # mat_i: Input matrix
    def pad_input(l):
      return l + [self._torch_tensor_apply(torch.tensor(0.0))] * (num_inputs - len(l)) if len(l) < num_inputs else l
    mat_i = torch.stack([torch.stack(pad_input(l)) for l in input_tags])

    # mat_w: Weight matrix
    mat_w = self._torch_tensor_apply((torch.zeros(batch_size, num_outputs, num_inputs)))
    for (batch_id, task_result) in enumerate(output_batch): # batch_size
      for (output_id, output_tag) in enumerate(task_result): # output_size
        if output_tag is not None:
          (_, deriv) = output_tag
          for (input_id, weight, _) in deriv:
            mat_w[batch_id, output_id, input_id] = weight

    # backward hook
    retain_graph = self.retain_graph
    def hook(grad):
      if mat_i.requires_grad:
        mat_f = torch.einsum("ikj,ik->ij", mat_w, grad)
        mat_i.backward(mat_f, retain_graph=retain_graph)

    return hook
