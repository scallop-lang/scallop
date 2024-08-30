import torch
import scallopy
import unittest

class TestSparseDigitForward(unittest.TestCase):
  def setUp(self):
    self.ctx = scallopy.ScallopContext(provenance="difftopkproofs")
    self.ctx.add_relation("digit_1", int, range(10))
    self.ctx.add_relation("digit_2", int, range(10))
    self.ctx.add_rule("sum_2(a + b) = digit_1(a) and digit_2(b)")
    self.ctx.add_rule("mult_2(a * b) = digit_1(a) and digit_2(b)")

  def test_backward_with_sparse(self):
    loss_fn = torch.nn.BCELoss()
    forward = self.ctx.forward_function(
      "sum_2",
      list(range(19)),
      sparse_jacobian=True)

    # Construct the digit
    digit_1_base = torch.randn((16, 10), requires_grad=True)
    digit_1 = torch.softmax(digit_1_base, dim=1)
    digit_2_base = torch.randn((16, 10), requires_grad=True)
    digit_2 = torch.softmax(digit_2_base, dim=1)

    # Call scallop and obtain loss
    sum_2 = forward(digit_1=digit_1, digit_2=digit_2)
    gt = torch.tensor([[1.0] + [0.0] * 18] * 16)
    l = loss_fn(sum_2, gt)

    # Ensure that there is no gradient
    assert digit_1_base.grad == None
    assert digit_2_base.grad == None

    # Perform backward
    l.backward()

    # Ensure that there is some gradient
    assert any(p != 0.0 for distr in digit_1_base.grad for p in distr)
    assert any(p != 0.0 for distr in digit_2_base.grad for p in distr)
