from typing import List

from . import torch_importer

class ScallopProvenance:
  """
  Base class for a provenance context. Any class implementing `ScallopProvenance`
  must override the following functions:
  - `zero`
  - `one`
  - `add`
  - `mult`
  - `negate`
  - `saturate`
  """

  def name(self):
    return "custom-provenance"

  def tagging_fn(self, input_tag):
    """
    Given the input tags, generate an internal tag for the tuple.
    For example, in the following code, internal tags are the I1, I2, ...:

    ``` python
    ctx.add_facts("RELA", [(I1, T1), (I2, T2), ...])
    ```

    This `tagging_fn` function should take in input tags like `I1` and return an internal tag

    If not implemented, we assume the input tags are the internal tags.
    If the input tag is not provided (`None`), we use the `one()` as the internal tag.
    """
    if input_tag is None: return self.one()
    else: return input_tag

  def recover_fn(self, internal_tag):
    """
    Given an internal tag, recover the output tag.

    If not implemented, we assume the internal tags are the output tags.
    """
    return internal_tag

  def discard(self, tag):
    """
    Check if a given tag needs to be discarded.
    When a tag is invalid, the tuple associated will be removed during reasoning.
    The default implementation assumes every tag is valid.

    An example of an invalid tag: a probability tag of probability 0.0
    """
    return False

  def zero(self):
    """
    Get the `0` element of the provenance semiring
    """
    raise Exception("Not implemented")

  def one(self):
    """
    Get the `1` element of the provenance semiring
    """
    raise Exception("Not implemented")

  def add(self, t1, t2):
    """
    Perform semiring addition on two tags (`t1` and `t2`)
    """
    raise Exception("Not implemented")

  def mult(self, t1, t2):
    """
    Perform semiring multiplication on two tags (`t1` and `t2`)
    """
    raise Exception("Not implemented")

  def negate(self, t):
    """
    Perform semiring negation on a tag (`t`)
    """
    raise Exception("Not implemented")

  def aggregate_count(self, elems):
    """
    Aggregate a count of the given elements
    """
    return [(self.one(), len(elems))]

  def aggregate_exists(self, elems):
    """
    Return whether there exists anything in the given elements
    """
    return [(self.one(), len(elems) > 0)]

  def aggregate_unique(self, elems):
    """
    Return a unique element from the given elements
    """
    if len(elems) > 0: return [elems[0]]
    else: return []

  def collate(self, batch_outputs):
    """
    Collate a List of List of tags into potentially a batched output
    """
    return batch_outputs


class DiffAddMultProb2Semiring(ScallopProvenance):
  """
  add-mult semiring: (R, 0.0, 1.0, +, *)

  This semiring uses PyTorch and expect differentiable probabilistic inputs
  as its tags.
  It will perform simple addition `+` and multiplication `*` when performing
  `OR` and `AND`.
  Note that when performing `+`, the probability can go over 1.0. We will
  perform a clamping in this case.
  It will not be probabilistically accurate but is very fast and may help
  during training.
  """
  def __init__(self):
    super(DiffAddMultProb2Semiring, self).__init__()
    if not torch_importer.has_pytorch:
      raise Exception("PyTorch unavailable. You can use this semiring only with PyTorch")

  def zero(self):
    """
    Zero tag is a floating point 0.0 (i.e. 0.0 probability being true)
    """
    return 0.0

  def one(self):
    """
    One tag is a floating point 1.0 (i.e. 1.0 probability being true)
    """
    return 1.0

  def add(self, a, b):
    """
    For a logical `OR` of probabilities `a` and `b`, the resulting tag is `clamp(a + b, 1.0)`
    """
    return torch_importer.torch.clamp(a + b, max=0.9999)

  def mult(self, a, b):
    """
    For a logical `AND` of probabilities `a` and `b`, the resulting tag is `a * b`
    """
    return a * b

  def collate(self, batch: List[List[torch_importer.Tensor]]):
    """
    Collate a batch of outputs
    """
    return torch_importer.torch.stack([
      torch_importer.torch.stack([
        p if p else torch_importer.torch.tensor(self.zero()) for p in outputs
      ]) for outputs in batch
    ])


class DiffNandMultProb2Semiring(ScallopProvenance):
  """
  Diff nand-mult semiring: (R, 0.0, 1.0, nand * not, *)

  This semiring uses PyTorch and expect differentiable probabilistic inputs
  as its tags.
  It will perform `1 - (1 - x)(1 - y)` for `OR` and `x * y` for `AND`.
  It will not be probabilistically accurate but is fairly fast and may help
  during training.
  """
  def __init__(self):
    super(DiffNandMultProb2Semiring, self).__init__()
    if not torch_importer.has_pytorch:
      raise Exception("PyTorch unavailable. You can use this semiring only with PyTorch")

  def zero(self):
    """
    Zero tag is a floating point 0.0 (i.e. 0.0 probability being true)
    """
    return 0.0

  def one(self):
    """
    One tag is a floating point 1.0 (i.e. 1.0 probability being true)
    """
    return 1.0

  def add(self, a, b):
    """
    Logical `OR(a, b)` is encoded as `NOT(AND(NOT(a), NOT(b)))`
    For a logical `OR` of probabilities `a` and `b`, the resulting tag is ``
    """
    return 1.0 - (1.0 - a) * (1.0 - b)

  def mult(self, a, b):
    """
    For a logical `AND` of probabilities `a` and `b`, the resulting tag is `a * b`
    """
    return a * b

  def collate(self, batch: List[List[torch_importer.Tensor]]):
    """
    Collate a batch of outputs
    """
    return torch_importer.torch.stack([torch_importer.torch.stack([p if p else torch_importer.torch.tensor(self.zero()) for p in outputs]) for outputs in batch])


class DiffMaxMultProb2Semiring(ScallopProvenance):
  """
  Diff max-mult semiring: (R, 0.0, 1.0, max, *)

  This semiring uses PyTorch and expect differentiable probabilistic inputs
  as its tags.
  It will perform simple max and multiplication when performing `OR` and `AND`.
  It will not be probabilistically accurate but is fairly fast and may help
  during training.
  """
  def __init__(self):
    super(DiffMaxMultProb2Semiring, self).__init__()
    if not torch_importer.has_pytorch:
      raise Exception("PyTorch unavailable. You can use this semiring only with PyTorch")

  def zero(self):
    """
    Zero tag is a floating point 0.0 (i.e. 0.0 probability being true)
    """
    return 0.0

  def one(self):
    """
    One tag is a floating point 1.0 (i.e. 1.0 probability being true)
    """
    return 1.0

  def add(self, a, b):
    """
    Logical `OR(a, b)` is encoded as `MAX(Pr(a), Pr(b))`
    """
    import torch
    return torch.max(a, b)

  def mult(self, a, b):
    """
    For a logical `AND(a, b)` is encoded as `Pr(a) * Pr(b)`
    """
    return a * b

  def collate(self, batch: List[List[torch_importer.Tensor]]):
    """
    Collate a batch of outputs
    """
    import torch
    return torch.stack([torch.stack([p if p else torch.tensor(self.zero()) for p in outputs]) for outputs in batch])
