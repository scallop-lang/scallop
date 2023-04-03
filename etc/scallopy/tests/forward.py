import unittest

import scallopy
import torch

class TestForward(unittest.TestCase):
  def test_top_k_sample(self):
    # First setup the symbols
    symbols = [str(s) for s in range(10)] + ["+", "-", "*", "/"]

    # Next setup the forward function
    ctx = scallopy.ScallopContext(provenance="difftopkproofs")
    ctx.add_relation("symbol", (int, str), [(i, s) for i in range(3) for s in symbols])
    ctx.add_rule("sampled_symbol(id, sym) :- sym = top<3>(s: symbol(id, s))")
    forward = ctx.forward_function("sampled_symbol")

    # Setup the input vectors
    predictions = torch.softmax(torch.randn((16, 3, 14)), dim=2)
    symbol = predictions.reshape((16, -1))
    (_, sampled_symbol_prob) = forward(symbol=symbol)

    # There should be 9 non-zeros in each row: 3 digits * top-3 per digit
    non_zeros = torch.count_nonzero(sampled_symbol_prob, dim=1)
    for x in non_zeros:
      assert int(x) == 9


class TestDigitForward(unittest.TestCase):
  def setUp(self):
    self.ctx = scallopy.ScallopContext(provenance="diffminmaxprob")
    self.ctx.add_relation("digit_1", int, range(10))
    self.ctx.add_relation("digit_2", int, range(10))
    self.ctx.add_rule("sum_2(a + b) = digit_1(a) and digit_2(b)")
    self.ctx.add_rule("mult_2(a * b) = digit_1(a) and digit_2(b)")

  @unittest.expectedFailure
  def test_unknown_relation_1(self):
    self.ctx.forward_function("add_3", list(range(28)))

  def test_normal(self):
    forward = self.ctx.forward_function("sum_2", list(range(19)))
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    sum_2 = forward(digit_1=digit_1, digit_2=digit_2)
    self.assertEqual(sum_2.shape, (16, 19))

  def test_no_output_mapping(self):
    forward = self.ctx.forward_function("sum_2")
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    (result_mapping, result_tensor) = forward(digit_1=digit_1, digit_2=digit_2)
    self.assertEqual(set(result_mapping), set([(i,) for i in range(19)]))
    self.assertEqual(result_tensor.shape, (16, 19))

  def test_single_dispatch(self):
    forward = self.ctx.forward_function("sum_2", dispatch="single")
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    (result_mapping, result_tensor) = forward(digit_1=digit_1, digit_2=digit_2)
    self.assertEqual(set(result_mapping), set([(i,) for i in range(19)]))
    self.assertEqual(result_tensor.shape, (16, 19))

  def test_serial_dispatch(self):
    forward = self.ctx.forward_function("sum_2", dispatch="serial")
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    (result_mapping, result_tensor) = forward(digit_1=digit_1, digit_2=digit_2)
    self.assertEqual(set(result_mapping), set([(i,) for i in range(19)]))
    self.assertEqual(result_tensor.shape, (16, 19))

  def test_multi_result(self):
    forward = self.ctx.forward_function(output_mappings={"sum_2": list(range(19)), "mult_2": list(range(100))})
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    result = forward(digit_1=digit_1, digit_2=digit_2)
    sum_2 = result["sum_2"]
    mult_2 = result["mult_2"]
    self.assertEqual(sum_2.shape, (16, 19))
    self.assertEqual(mult_2.shape, (16, 100))

  def test_multi_result_single_dispatch(self):
    forward = self.ctx.forward_function(output_mappings={"sum_2": list(range(19)), "mult_2": list(range(100))}, dispatch="single")
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    result = forward(digit_1=digit_1, digit_2=digit_2)
    sum_2 = result["sum_2"]
    mult_2 = result["mult_2"]
    self.assertEqual(sum_2.shape, (16, 19))
    self.assertEqual(mult_2.shape, (16, 100))

  def test_multi_result_non_parallel_dispatch(self):
    forward = self.ctx.forward_function(output_mappings={"sum_2": list(range(19)), "mult_2": list(range(100))}, dispatch="serial")
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    result = forward(digit_1=digit_1, digit_2=digit_2)
    sum_2 = result["sum_2"]
    mult_2 = result["mult_2"]
    self.assertEqual(sum_2.shape, (16, 19))
    self.assertEqual(mult_2.shape, (16, 100))

  def test_multi_result_maybe_with_output_mapping(self):
    forward = self.ctx.forward_function(output_mappings={"sum_2": None, "mult_2": list(range(100))})
    digit_1 = torch.softmax(torch.randn((16, 10)), dim=1)
    digit_2 = torch.softmax(torch.randn((16, 10)), dim=1)
    result = forward(digit_1=digit_1, digit_2=digit_2)
    (sum_2_mapping, sum_2_tensor) = result["sum_2"]
    mult_2 = result["mult_2"]
    self.assertEqual(set(sum_2_mapping), set([(i,) for i in range(19)]))
    self.assertEqual(sum_2_tensor.shape, (16, 19))
    self.assertEqual(mult_2.shape, (16, 100))


class TestDigitForwardWithJIT(unittest.TestCase):
  def setUp(self):
    self.ctx = scallopy.ScallopContext(provenance="diffminmaxprob")
    self.ctx.add_relation("digit_1", int, list(range(10)))
    self.ctx.add_relation("digit_2", int, list(range(10)))
    self.ctx.add_rule("sum_2(a + b) = digit_1(a) and digit_2(b)")
    self.ctx.add_rule("mult_2(a * b) = digit_1(a) and digit_2(b)")

  @unittest.expectedFailure
  def test_unknown_relation(self):
    self.ctx.forward_function("add_3", list(range(28)))

  @unittest.expectedFailure
  def test_no_ouputs(self):
    self.ctx.forward_function(output_mappings={}, jit=True)

  def test_normal(self):
    forward = self.ctx.forward_function("sum_2", list(range(19)), jit=True, jit_name="digit")
    digit_1, digit_2 = torch.randn((16, 10)), torch.randn((16, 10))
    result = forward(digit_1=digit_1, digit_2=digit_2)
    self.assertEqual(result.shape, (16, 19))

  def test_normal_single_dispatch(self):
    forward = self.ctx.forward_function("sum_2", list(range(19)), jit=True, jit_name="digit", dispatch="single")
    digit_1, digit_2 = torch.randn((16, 10)), torch.randn((16, 10))
    result = forward(digit_1=digit_1, digit_2=digit_2)
    self.assertEqual(result.shape, (16, 19))

  def test_normal_non_parallel_dispatch(self):
    forward = self.ctx.forward_function("sum_2", list(range(19)), jit=True, jit_name="digit", dispatch="serial")
    digit_1, digit_2 = torch.randn((16, 10)), torch.randn((16, 10))
    result = forward(digit_1=digit_1, digit_2=digit_2)
    self.assertEqual(result.shape, (16, 19))

  def test_multi_result_normal(self):
    forward = self.ctx.forward_function(output_mappings={"sum_2": list(range(19)), "mult_2": list(range(100))}, jit=True, jit_name="digit_multi_result")
    digit_1, digit_2 = torch.randn((16, 10)), torch.randn((16, 10))
    result = forward(digit_1=digit_1, digit_2=digit_2)
    self.assertEqual(result["sum_2"].shape, (16, 19))
    self.assertEqual(result["mult_2"].shape, (16, 100))

  def test_multi_result_single_dispatch(self):
    forward = self.ctx.forward_function(output_mappings={"sum_2": list(range(19)), "mult_2": list(range(100))}, jit=True, jit_name="digit_multi_result", dispatch="single")
    digit_1, digit_2 = torch.randn((16, 10)), torch.randn((16, 10))
    result = forward(digit_1=digit_1, digit_2=digit_2)
    self.assertEqual(result["sum_2"].shape, (16, 19))
    self.assertEqual(result["mult_2"].shape, (16, 100))

  def test_multi_result_non_parallel_dispatch(self):
    forward = self.ctx.forward_function(output_mappings={"sum_2": list(range(19)), "mult_2": list(range(100))}, jit=True, jit_name="digit_multi_result", dispatch="serial")
    digit_1, digit_2 = torch.randn((16, 10)), torch.randn((16, 10))
    result = forward(digit_1=digit_1, digit_2=digit_2)
    self.assertEqual(result["sum_2"].shape, (16, 19))
    self.assertEqual(result["mult_2"].shape, (16, 100))


class TestDirectForward(unittest.TestCase):
  def test_forward(self):
    sum_2_program = """
    type digit_a(usize), digit_b(usize)
    rel sum_2(a + b) = digit_a(a), digit_b(b)
    """
    compute_sum_2 = scallopy.ScallopForwardFunction(
      program=sum_2_program,
      provenance="difftopkproofs",
      input_mappings={"digit_a": list(range(10)), "digit_b": list(range(10))},
      output_mappings={"sum_2": list(range(19))},
    )
    digit_a, digit_b = torch.randn((16, 10)), torch.randn((16, 10))
    result = compute_sum_2(digit_a=digit_a, digit_b=digit_b)
    self.assertEqual(result.shape, (16, 19))

  def test_forward_with_probabilities(self):
    def process_input(digit_tensor):
      r = []
      (batch_size, _) = digit_tensor.shape
      for task_id in range(batch_size):
        r.append([(p, (i,)) for (i, p) in enumerate(digit_tensor[task_id])])
      return r

    sum_2_program = """
    type digit_a(usize), digit_b(usize)
    rel sum_2(a + b) = digit_a(a), digit_b(b)
    """
    compute_sum_2 = scallopy.ScallopForwardFunction(
      program=sum_2_program,
      provenance="difftopkproofs",
      input_mappings={"digit_a": None, "digit_b": None},
      output_mappings={"sum_2": list(range(19))},
    )
    digit_a, digit_b = torch.randn((16, 10)), torch.randn((16, 10))
    result = compute_sum_2(digit_a=process_input(digit_a), digit_b=process_input(digit_b))
    self.assertEqual(result.shape, (16, 19))

  def test_forward_with_non_probabilistic(self):
    edge_path_program = """
    type edge(usize, usize)
    rel path(a, c) = edge(a, c) or (path(a, b) and edge(b, c))
    """
    compute_path = scallopy.ScallopForwardFunction(
      program=edge_path_program,
      provenance="difftopkproofs",
      non_probabilistic=["edge"],
      output_relation="path")
    edges = [[(0, 1), (1, 2)]]
    output_mapping, result = compute_path(edge=edges)
    self.assertEqual(set(output_mapping), set([(0, 1), (0, 2), (1, 2)]))
    self.assertEqual(result.shape, (1, 3))

if __name__ == "__main__":
  unittest.main()
