from typing import Tuple

from . import value_types
from .predicate import foreign_predicate, Generator

@foreign_predicate(name="soft_eq", type_params=[value_types.Tensor])
def soft_eq_tensor(x: value_types.Tensor, y: value_types.Tensor) -> Generator[value_types.Tensor, Tuple]:
  import torch
  cs = torch.cosine_similarity(x, y, dim=0)
  prob = cs + 1.0 / 2.0
  yield (prob, ())

@foreign_predicate(name="soft_neq", type_params=[value_types.Tensor])
def soft_neq_tensor(x: value_types.Tensor, y: value_types.Tensor) -> Generator[value_types.Tensor, Tuple]:
  import torch
  cs = torch.cosine_similarity(x, y, dim=0)
  prob = 1.0 - (cs + 1.0 / 2.0)
  yield (prob, ())

STDLIB = {
  "functions": [
    # TODO
  ],
  "predicates": [
    soft_eq_tensor,
    soft_neq_tensor,
  ],
  "attributes": [
    # TODO
  ],
}
