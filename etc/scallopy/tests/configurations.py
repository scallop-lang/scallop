import unittest

import scallopy

class ConfigurationTests(unittest.TestCase):
  def test_iter_limit_1(self):
    ctx = scallopy.ScallopContext()
    ctx.set_iter_limit(2)
    ctx.add_relation("edge", (int, int))
    ctx.add_facts("edge", [(0, 1), (1, 2), (2, 3), (3, 4)])
    ctx.add_rule("path(a, c) = edge(a, c) or path(a, b) and edge(b, c)")
    ctx.run()
    assert list(ctx.relation("path")) == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]

  def test_iter_limit_2(self):
    ctx = scallopy.ScallopContext()
    ctx.set_iter_limit(1)
    ctx.set_iter_limit(None)
    ctx.add_relation("edge", (int, int))
    ctx.add_facts("edge", [(0, 1), (1, 2), (2, 3)])
    ctx.add_rule("path(a, c) = edge(a, c) or path(a, b) and edge(b, c)")
    ctx.run()
    assert list(ctx.relation("path")) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

  def test_hidden_relation_1(self):
    ctx = scallopy.ScallopContext()
    ctx.add_program(
      """
      @hidden
      type a(i32, i32)
      type b(i32, i32, i32)
      """
    )
    self.assertEqual(ctx.relations(), ["b"])
