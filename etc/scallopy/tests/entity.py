import unittest

import scallopy

class TestEntity(unittest.TestCase):
  def test_entity_1(self):
    ctx = scallopy.Context()
    ctx.add_program("type Expr = Const(i32) | Add(Expr, Expr)")
    ctx.add_relation("root", scallopy.Entity)
    ctx.add_rule("eval(e, y) = case e is Const(y)")
    ctx.add_rule("eval(e, y1 + y2) = case e is Add(e1, e2) and eval(e1, y1) and eval(e2, y2)")
    ctx.add_rule("result(y) = root(e) and eval(e, y)")
    ctx.add_facts("root", [("Add(Const(5), Add(Const(3), Const(4)))",)])
    ctx.run()
    assert list(ctx.relation("result")) == [(12,)]

  def test_entity_2(self):
    ctx = scallopy.Context()
    ctx.add_program("""
      type Expr = Const(i32) | Add(Expr, Expr)
      type root(expr: Expr)
      rel eval(e, y) = case e is Const(y)
      rel eval(e, y1 + y2) = case e is Add(e1, e2) and eval(e1, y1) and eval(e2, y2)
      rel result(y) = root(e) and eval(e, y)
    """)
    ctx.add_facts("root", [("Add(Const(5), Add(Const(3), Const(4)))",)])
    ctx.run()
    assert list(ctx.relation("result")) == [(12,)]

  def test_entity_tuple_1(self):
    ctx = scallopy.Context()
    ctx.add_program("""
      type Expr = Const(i32) | Add(Expr, Expr)
      type target(id: i32, b: Expr)
      rel eval(e, y) = case e is Const(y)
      rel eval(e, y1 + y2) = case e is Add(e1, e2) and eval(e1, y1) and eval(e2, y2)
      rel result(id, y) = target(id, e) and eval(e, y)
    """)
    ctx.add_facts("target", [(1, "Add(Const(3), Const(5))"), (2, "Add(Const(6), Const(5))")])
    ctx.run()
    assert list(ctx.relation("result")) == [(1, 8), (2, 11)]

  def test_entity_compile_failure_1(self):
    # Unexisted variant
    ctx = scallopy.Context()
    ctx.add_relation("root", scallopy.Entity)
    ctx.add_facts("root", [("Unexisted(5)",)])
    ctx.run()
    assert len(list(ctx.relation("root"))) == 0

  def test_entity_compile_failure_2(self):
    # Arity mismatch
    ctx = scallopy.Context()
    ctx.add_program("type Expr = Const(i32) | Add(Expr, Expr)")
    ctx.add_relation("root", scallopy.Entity)
    ctx.add_facts("root", [("Add(Const(5), Const(3), Const(7))",)])
    ctx.run()
    assert len(list(ctx.relation("root"))) == 0

  def test_entity_forward(self):
    forward = scallopy.Module(
      program="""
        type Expr = Const(i32) | Add(Expr, Expr)
        type root(expr: Expr)
        rel eval(e, y) = case e is Const(y)
        rel eval(e, y1 + y2) = case e is Add(e1, e2) and eval(e1, y1) and eval(e2, y2)
        rel result(y) = root(e) and eval(e, y)
      """,
      provenance="diffminmaxprob",
      non_probabilistic=["root"],
      output_relation="result")

    results, _ = forward(
      root=[
        [("Add(Const(5), Const(3))",)],
        [("Const(10)",)],
        [("Add(Add(Const(10), Const(2)), Const(2))",)],
      ]
    )

    assert set(results) == set([(8,), (10,), (14,)])

  def test_entity_forward_2(self):
    forward = scallopy.Module(
      program="""
        type Expr = Const(i32) | Add(Expr, Expr)
        type root(id: i32, expr: Expr)
        rel eval(e, y) = case e is Const(y)
        rel eval(e, y1 + y2) = case e is Add(e1, e2) and eval(e1, y1) and eval(e2, y2)
        rel result(id, y) = root(id, e) and eval(e, y)
      """,
      provenance="diffminmaxprob",
      non_probabilistic=["root"],
      output_relation="result")

    results, _ = forward(
      root=[
        [(1, "Add(Const(5), Const(3))"), (2, "Const(3)")],
        [(1, "Const(10)"), (2, "Add(Add(Const(1), Const(2)), Const(3))")],
        [(1, "Add(Add(Const(10), Const(2)), Const(2))")],
      ]
    )

    assert set(results) == set([(1, 8), (1, 10), (1, 14), (2, 3), (2, 6)])
