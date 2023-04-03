import unittest
import torch

import scallopy

class TestInputMapping(unittest.TestCase):
  def test_construct_list_1(self):
    im = scallopy.InputMapping(list(range(10)))
    assert im.kind == "list"
    assert im.shape == (10,)
    assert im.dimension == 1
    assert im.is_singleton == True

  def test_construct_list_1_1(self):
    im = scallopy.InputMapping(range(10))
    assert im.kind == "list"
    assert im.shape == (10,)
    assert im.dimension == 1
    assert im.is_singleton == True

  def test_construct_list_2(self):
    im = scallopy.InputMapping([(i,) for i in range(10)])
    assert im.kind == "list"
    assert im.shape == (10,)
    assert im.dimension == 1
    assert im.is_singleton == False

  def test_construct_list_3(self):
    im = scallopy.InputMapping([(i, j) for i in range(5) for j in range(5)])
    assert im.kind == "list"
    assert im.shape == (25,)
    assert im.dimension == 1
    assert im.is_singleton == False

  def test_construct_list_4(self):
    im = scallopy.InputMapping([[(i, j) for i in range(5)] for j in range(5)])
    assert im.kind == "list"
    assert im.shape == (5, 5)
    assert im.dimension == 2
    assert im.is_singleton == False

  @unittest.expectedFailure
  def test_construct_list_failure_1(self):
    # Tuple size mismatch
    _ = scallopy.InputMapping([(1, 2), (1,)])

  @unittest.expectedFailure
  def test_construct_list_failure_2(self):
    # Empty mapping
    _ = scallopy.InputMapping([])

  @unittest.expectedFailure
  def test_construct_list_failure_3(self):
    # Empty mapping on other dimensions
    _ = scallopy.InputMapping([[], [], []])

  @unittest.expectedFailure
  def test_construct_list_failure_4(self):
    # Unmatched dimensions
    _ = scallopy.InputMapping([[3, 5, 8], [3, 5, 9], [3, 5]])

  @unittest.expectedFailure
  def test_cannot_set_property(self):
    im = scallopy.InputMapping(range(10))
    im.dimension = (1, 1, 1, 1)

  def test_construct_tuple_1(self):
    im = scallopy.InputMapping(())
    assert im.kind == "tuple"
    assert im.shape == ()
    assert im.dimension == 0
    assert im.is_singleton == False

  def test_construct_tuple_2(self):
    im = scallopy.InputMapping((3, 5))
    assert im.kind == "tuple"
    assert im.shape == ()
    assert im.dimension == 0
    assert im.is_singleton == False

  @unittest.expectedFailure
  def test_construct_tuple_failure_1(self):
    _ = scallopy.InputMapping((3, 5, []))

  def test_construct_value_1(self):
    im = scallopy.InputMapping(3)
    assert im.kind == "value"
    assert im.shape == ()
    assert im.dimension == 0
    assert im.is_singleton == True

  def test_construct_dict_1(self):
    im = scallopy.InputMapping({0: range(5), 1: range(5)})
    assert im.kind == "dict"
    assert im.shape == (5, 5)
    assert im.dimension == 2
    assert im.is_singleton == False

  def test_construct_dict_2(self):
    im = scallopy.InputMapping({0: range(5), 1: range(5), 2: range(2)})
    assert im.kind == "dict"
    assert im.shape == (5, 5, 2)
    assert im.dimension == 3
    assert im.is_singleton == False

  def test_construct_dict_3(self):
    im = scallopy.InputMapping({0: range(3), 1: ["red", "green", "blue"]})
    assert im.kind == "dict"
    assert im.shape == (3, 3)
    assert im.dimension == 2
    assert im.is_singleton == False

  @unittest.expectedFailure
  def test_construct_dict_failure_1(self):
    _ = scallopy.InputMapping({})

  @unittest.expectedFailure
  def test_construct_dict_failure_2(self):
    _ = scallopy.InputMapping({1: range(3)})

  @unittest.expectedFailure
  def test_construct_dict_failure_3(self):
    _ = scallopy.InputMapping({0: range(3), 1: []})

  @unittest.expectedFailure
  def test_construct_dict_failure_4(self):
    _ = scallopy.InputMapping({0: [(1, 3, 5)]})

  @unittest.expectedFailure
  def test_construct_dict_failure_5(self):
    _ = scallopy.InputMapping({"1": [3]})

  @unittest.expectedFailure
  def test_construct_dict_failure_6(self):
    _ = scallopy.InputMapping({-10: [3]})

  def test_process_tensor_1(self):
    im = scallopy.InputMapping({0: range(5), 1: range(5)})
    r = im.process_tensor(torch.zeros((5, 5)))
    assert len(r) == 25

  def test_process_tensor_2(self):
    im = scallopy.InputMapping([[(i, j) for j in range(5)] for i in range(5)])
    r = im.process_tensor(torch.zeros((5, 5)), batched=True)
    assert len(r) == 1
    assert len(r[0]) == 25

  def test_process_tensor_3(self):
    im = scallopy.InputMapping(range(10))
    r = im.process_tensor(torch.randn((16, 10)))
    assert len(r) == 16
    assert len(r[0]) == 10

  def test_retain_k_1(self):
    im = scallopy.InputMapping(range(10), retain_k=3)
    r = im.process_tensor(torch.randn((10,)))
    assert len(r) == 3

  @unittest.expectedFailure
  def test_retain_k_2(self):
    _ = scallopy.InputMapping(range(10), retain_k=3, sample_dim=1)

  @unittest.expectedFailure
  def test_retain_k_3(self):
    _ = scallopy.InputMapping(range(10), retain_k=3, sample_dim=-10)

  def test_mult_dim_retain_k_1(self):
    im = scallopy.InputMapping({0: range(5), 1: range(5)}, retain_k=3)
    r = im.process_tensor(torch.randn((5, 5)))
    assert len(r) == 3

  def test_mult_dim_retain_k_2(self):
    im = scallopy.InputMapping({0: range(5), 1: range(3)}, retain_k=2, sample_dim=1)
    r = im.process_tensor(torch.randn((5, 3)))
    assert len(r) == 10

  def test_mult_dim_retain_k_3(self):
    im = scallopy.InputMapping({0: range(5), 1: range(3)}, retain_k=2, sample_dim=0)
    r = im.process_tensor(torch.randn((5, 3)))
    assert len(r) == 6

  def test_retain_threshold_1(self):
    im = scallopy.InputMapping(range(10), retain_threshold=0.5)
    t = torch.randn(10)
    r = im.process_tensor(t)
    assert len(r) == len(t[t > 0.5])

  def test_retain_threshold_2(self):
    im = scallopy.InputMapping({0: range(5), 1: range(3)}, retain_threshold=0.5)
    t = torch.randn((5, 3))
    r = im.process_tensor(t)
    assert len(r) == len(t[t > 0.5])

  def test_disjunction_1(self):
    im = scallopy.InputMapping(range(10), disjunctive=True, supports_disjunctions=True)
    t = torch.randn((10,))
    r = im.process_tensor(t)
    for ((_, did), _) in r:
      assert did == 0

  def test_disjunction_2(self):
    im = scallopy.InputMapping({0: range(5), 1: range(5)}, disjunctive=True, supports_disjunctions=True)
    t = torch.randn((5, 5))
    r = im.process_tensor(t)
    for ((_, did), _) in r:
      assert did == 0

  def test_disjunction_3(self):
    im = scallopy.InputMapping({0: range(5), 1: range(5)}, disjunctive_dim=1, supports_disjunctions=True)
    t = torch.randn((5, 5))
    r = im.process_tensor(t)
    for ((_, did), (i, _)) in r:
      assert did == i

  def test_disjunction_4(self):
    im = scallopy.InputMapping({0: range(5), 1: range(5)}, disjunctive_dim=0, supports_disjunctions=True)
    t = torch.randn((5, 5))
    r = im.process_tensor(t)
    for ((_, did), (_, j)) in r:
      assert did == j
