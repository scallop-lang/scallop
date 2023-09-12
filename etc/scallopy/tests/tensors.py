import unittest

import torch
import scallopy

class TensorTests(unittest.TestCase):
  @unittest.skipIf(not scallopy.torch_tensor_enabled(), "not supported in this scallopy version")
  def test_tensor_1(self):
    x, y = torch.randn(5), torch.randn(5)
    s = x.dot(y)

    ctx = scallopy.Context()
    ctx.add_relation("r", (int, scallopy.Tensor))
    ctx.add_facts("r", [(1, x), (2, y)])
    ctx.add_rule("y($dot(ta, tb)) = r(1, ta) and r(2, tb)")

    ctx.run()

    result = list(ctx.relation("y"))[0][0]
    assert s == result

  @unittest.skipIf(not scallopy.torch_tensor_enabled(), "not supported in this scallopy version")
  def test_tensor_2(self):
    x, y = torch.randn(5), torch.randn(5)
    gt_sum = x + y

    ctx = scallopy.Context()
    ctx.add_relation("r", (int, scallopy.Tensor))
    ctx.add_facts("r", [(1, x), (2, y)])
    ctx.add_rule("y(ta + tb) = r(1, ta) and r(2, tb)")
    ctx.run()
    my_sum = list(ctx.relation("y"))[0][0]

    assert all(gt_sum == my_sum)

  @unittest.skipIf(not scallopy.torch_tensor_enabled(), "not supported in this scallopy version")
  def test_tensor_3(self):
    x = torch.randn(10)
    y = torch.randn(10)
    gt_sim = x.dot(y) / (x.norm() * y.norm()) + 1.0 / 2.0

    ctx = scallopy.Context(provenance="difftopkproofs")
    ctx.add_relation("embed", (int, scallopy.Tensor), non_probabilistic=True)
    ctx.add_facts("embed", [(1, x), (2, y)])
    ctx.add_rule("similar(x, y) = embed(x, tx) and embed(y, ty) and x != y and soft_eq<Tensor>(tx, ty)")
    ctx.run()
    my_sim = list(ctx.relation("similar"))[0][0]

    assert abs(gt_sim.item() - my_sim.item()) < 0.001

  @unittest.skipIf(not scallopy.torch_tensor_enabled(), "not supported in this scallopy version")
  def test_tensor_backprop_4(self):
    x = torch.randn(10, requires_grad=True)
    y = torch.randn(10)
    opt = torch.optim.Adam(params=[x], lr=0.1)
    gt_initial_sim = x.dot(y) / (x.norm() * y.norm()) + 1.0 / 2.0

    ctx = scallopy.Context(provenance="difftopkproofs")
    ctx.add_relation("embed", (int, scallopy.Tensor), non_probabilistic=True)
    ctx.add_facts("embed", [(1, x), (2, y)])
    ctx.add_rule("similar(x, y) = embed(x, tx) and embed(y, ty) and x != y and soft_eq<Tensor>(tx, ty)")
    ctx.run()
    my_initial_sim = list(ctx.relation("similar"))[0][0]

    assert abs(gt_initial_sim.item() - my_initial_sim.item()) < 0.001

    # Derive a loss, backward, and step
    l = torch.nn.functional.mse_loss(my_initial_sim, torch.tensor(1.0))
    l.backward()
    opt.step()

    # New similarity
    new_sim = x.dot(y) / (x.norm() * y.norm()) + 1.0 / 2.0
    assert new_sim > my_initial_sim

  @unittest.skipIf(not scallopy.torch_tensor_enabled(), "not supported in this scallopy version")
  def test_tensor_forward_backprop_1(self):
    batch_size = 16

    x = torch.randn((batch_size, 10), requires_grad=True)
    y = torch.randn((batch_size, 10), requires_grad=True)
    opt = torch.optim.Adam(params=[x, y], lr=0.1)

    scl_module = scallopy.Module(
      program="""
        type embedding_1(embed: Tensor)
        type embedding_2(embed: Tensor)
        rel similar() = embedding_1(t1) and embedding_2(t2) and soft_eq<Tensor>(t1, t2)
        query similar
      """,
      non_probabilistic=["embedding_1", "embedding_2"],
      output_relation="similar",
      output_mapping=(),
      dispatch="serial")

    def step() -> float:
      result = scl_module(embedding_1=[[(x[i],)] for i in range(batch_size)], embedding_2=[[(y[i],)] for i in range(batch_size)])
      gt = torch.ones(batch_size)
      l = torch.nn.functional.mse_loss(result, gt)
      l.backward()
      opt.step()
      return l.item()

    curr_loss = step()
    for i in range(4):
      next_loss = step()
      assert next_loss < curr_loss
      curr_loss = next_loss
