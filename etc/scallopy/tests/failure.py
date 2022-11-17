import unittest

import scallopy

class TestFailures(unittest.TestCase):
  @unittest.expectedFailure
  def test_type_error_1(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("r", (int, int))
    ctx.add_facts("r", [(1, 3), ("wrong", "type")])

  @unittest.expectedFailure
  def test_type_error_2(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("r", ("usize", int))
    ctx.add_facts("r", [(-1, 3), (5, 6)])

  @unittest.expectedFailure
  def test_no_relation_1(self):
    ctx = scallopy.ScallopContext()
    ctx.add_facts("r", [(-1, 3), (5, 6)])

  @unittest.expectedFailure
  def test_incorrect_input_mapping_1(self):
    ctx = scallopy.ScallopContext(provenance="difftopkproofs")
    ctx.add_relation("digit", (int, int), input_mapping=list(range(10)))

  @unittest.expectedFailure
  def test_incorrect_input_mapping_2(self):
    ctx = scallopy.ScallopContext(provenance="difftopkproofs")
    ctx.set_input_mapping("digit", list(range(10)))

  @unittest.expectedFailure
  def test_incorrect_input_mapping_3(self):
    ctx = scallopy.ScallopContext(provenance="difftopkproofs")
    ctx.add_relation("digit", (int, int))
    ctx.set_input_mapping("digit", list(range(10)))

  @unittest.expectedFailure
  def test_incorrect_output_mapping_1(self):
    ctx = scallopy.ScallopContext(provenance="difftopkproofs")
    ctx.add_relation("digit_1", int, input_mapping=list(range(10)))
    ctx.add_relation("digit_2", int, input_mapping=list(range(10)))
    ctx.add_rule("sum2(a + b) = digit_1(a), digit_2(b)")
    ctx.forward_function("sum2", [(i, i + 1) for i in range(19)])


if __name__ == "__main__":
  unittest.main()
