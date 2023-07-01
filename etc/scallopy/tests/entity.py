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
    ctx.add_entity("root", "Add(Const(5), Add(Const(3), Const(4)))")
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
    ctx.add_entity("root", "Add(Const(5), Add(Const(3), Const(4)))")
    ctx.run()
    assert list(ctx.relation("result")) == [(12,)]

  def test_entity_constant_1(self):
    ctx = scallopy.Context()
    ctx.add_program("""type root(b: bool)""")
    ctx.add_entity("root", True)
    ctx.run()
    assert list(ctx.relation("root")) == [(True,)]

  def test_entity_constant_2(self):
    ctx = scallopy.Context()
    ctx.add_program("type root(b: String)")
    ctx.add_entity("root", "hello world")
    ctx.run()
    assert list(ctx.relation("root")) == [("hello world",)]

  def test_entity_constant_3(self):
    ctx = scallopy.Context()
    ctx.add_program("type root(b: i32)")
    ctx.add_entity("root", 3)
    ctx.run()
    assert list(ctx.relation("root")) == [(3,)]

  def test_entity_tuple_1(self):
    ctx = scallopy.Context()
    ctx.add_program("""
      type Expr = Const(i32) | Add(Expr, Expr)
      type root(id: i32, b: Expr)
      rel eval(e, y) = case e is Const(y)
      rel eval(e, y1 + y2) = case e is Add(e1, e2) and eval(e1, y1) and eval(e2, y2)
      rel result(id, y) = root(id, e) and eval(e, y)
    """)
    ctx.add_entity("root", (1, "Add(Const(3), Const(5))"))
    ctx.add_entity("root", (2, "Add(Const(6), Const(5))"))
    ctx.run()
    assert list(ctx.relation("result")) == [(1, 8), (2, 11)]

  @unittest.expectedFailure
  def test_entity_compile_failure_1(self):
    # Unexisted variant
    ctx = scallopy.Context()
    ctx.add_relation("root", scallopy.Entity)
    ctx.add_entity("root", "Unexisted(5)")

  @unittest.expectedFailure
  def test_entity_compile_failure_2(self):
    # Arity mismatch
    ctx = scallopy.Context()
    ctx.add_program("type Expr = Const(i32) | Add(Expr, Expr)")
    ctx.add_relation("root", scallopy.Entity)
    ctx.add_entity("root", "Add(Const(5), Const(3), Const(7))")

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
      output_relation="result")

    results, _ = forward(
      entities={
        "root": [
          ["Add(Const(5), Const(3))"],
          ["Const(10)"],
          ["Add(Add(Const(10), Const(2)), Const(2))"],
        ]
      }
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
      output_relation="result")

    results, _ = forward(
      entities={
        "root": [
          [(1, "Add(Const(5), Const(3))"), (2, "Const(3)")],
          [(1, "Const(10)"), (2, "Add(Add(Const(1), Const(2)), Const(3))")],
          [(1, "Add(Add(Const(10), Const(2)), Const(2))")],
        ]
      }
    )

    assert set(results) == set([(1, 8), (1, 10), (1, 14), (2, 3), (2, 6)])
