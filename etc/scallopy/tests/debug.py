import unittest
import torch

import scallopy

class TestDebugProvenance(unittest.TestCase):
  def test_debug_1(self):
    ctx = scallopy.ScallopContext(provenance="difftopkproofsdebug")
    ctx.add_relation("digit_a", int)
    ctx.add_relation("digit_b", int)
    ctx.add_rule("sum_2(a + b) = digit_a(a) and digit_b(b)")
    ctx.add_facts("digit_a", [((torch.tensor(0.1), 1), (1,)), ((torch.tensor(0.9), 2), (2,))])
    ctx.add_facts("digit_b", [((torch.tensor(0.9), 3), (1,)), ((torch.tensor(0.1), 4), (2,))])
    ctx.run()
    result = ctx.relation("sum_2")
    for ((prob, proofs), (summation,)) in result:
      if summation == 2:
        self.assertAlmostEqual(float(prob), 0.09)
        self.assertEqual(proofs, [[(True, 1), (True, 3)]])
      elif summation == 3:
        self.assertAlmostEqual(float(prob), 0.82, 1)
        self.assertEqual(proofs, [[(True, 2), (True, 3)], [(True, 1), (True, 4)]])
      elif summation == 4:
        self.assertAlmostEqual(float(prob), 0.09)
        self.assertEqual(proofs, [[(True, 2), (True, 4)]])

  def test_debug_forward_1(self):
    fn = scallopy.Module(
      provenance="difftopkproofsdebug",
      program="""
        type digit_a(a: i32), digit_b(b: i32)
        rel sum_2(a + b) = digit_a(a) and digit_b(b)
      """,
      output_relation="sum_2",
    )

    digit_a = [
      [((torch.tensor(0.1), 1), (1,)), ((torch.tensor(0.9), 2), (2,))], # Datapoint 1
    ]
    digit_b = [
      [((torch.tensor(0.9), 3), (1,)), ((torch.tensor(0.1), 4), (2,))], # Datapoint 1
    ]

    (mapping, result_tensor, proofs) = fn(digit_a=digit_a, digit_b=digit_b)

    for (i, (result,)) in enumerate(mapping):
      if result == 2:
        self.assertAlmostEqual(float(result_tensor[0][i]), 0.09)
        self.assertEqual(proofs[0][i], [[(True, 1), (True, 3)]])
      if result == 3:
        self.assertAlmostEqual(float(result_tensor[0][i]), 0.8119, 3)
        self.assertEqual(proofs[0][i], [[(True, 2), (True, 3)], [(True, 1), (True, 4)]])
      if result == 4:
        self.assertAlmostEqual(float(result_tensor[0][i]), 0.09)
        self.assertEqual(proofs[0][i], [[(True, 2), (True, 4)]])

  def test_debug_forward_2(self):
    fn = scallopy.Module(
      provenance="difftopkproofsdebug",
      program="""
        type digit_a(a: i32), digit_b(b: i32)
        rel sum_2(a + b) = digit_a(a) and digit_b(b)
      """,
      output_relation="sum_2",
      output_mapping=[2, 3, 4],
    )

    digit_a = [
      [((torch.tensor(0.1), 1), (1,)), ((torch.tensor(0.9), 2), (2,))], # Datapoint 1
    ]
    digit_b = [
      [((torch.tensor(0.9), 3), (1,)), ((torch.tensor(0.1), 4), (2,))], # Datapoint 1
    ]

    (result_tensor, proofs) = fn(digit_a=digit_a, digit_b=digit_b)

    self.assertAlmostEqual(float(result_tensor[0][0]), 0.09)
    self.assertEqual(proofs[0][0], [[(True, 1), (True, 3)]])
    self.assertAlmostEqual(float(result_tensor[0][1]), 0.8119, 3)
    self.assertEqual(proofs[0][1], [[(True, 2), (True, 3)], [(True, 1), (True, 4)]])
    self.assertAlmostEqual(float(result_tensor[0][2]), 0.09)
    self.assertEqual(proofs[0][2], [[(True, 2), (True, 4)]])

  @unittest.expectedFailure
  def test_debug_forward_error_1(self):
    fn = scallopy.Module(
      provenance="difftopkproofsdebug",
      program="""
        type digit_a(a: i32), digit_b(b: i32)
        rel sum_2(a + b) = digit_a(a) and digit_b(b)
      """,
      input_mappings={
        "digit_a": range(10),
        "digit_b": range(10),
      },
      output_relation="sum_2",
    )

    digit_a = torch.randn((10,))
    digit_b = torch.randn((10,))
    result = fn(digit_a=digit_a, digit_b=digit_b)

if __name__ == "__main__":
  unittest.main()
