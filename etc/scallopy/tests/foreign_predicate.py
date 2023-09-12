from typing import *
import unittest

import scallopy

class TestForeignPredicate(unittest.TestCase):
  def test_foreign_predicate_range(self):
    @scallopy.foreign_predicate
    def my_range(a: int, b: int) -> scallopy.Generator[None, Tuple[int]]:
      for i in range(a, b):
        yield (i,)

    ctx = scallopy.ScallopContext()
    ctx.register_foreign_predicate(my_range)
    ctx.add_rule("r(x) = my_range(1, 5, x)")
    ctx.run()
    self.assertEqual(list(ctx.relation("r")), [(1,), (2,), (3,), (4,)])

  def test_fp_entity(self):
    @scallopy.foreign_predicate
    def my_dummy_semantic_parser(s: str) -> scallopy.Generator[None, Tuple[scallopy.Entity]]:
      if s == "If I have 3 apples and 2 pears, how many fruits do I have?":
        yield ("Add(Const(3), Const(2))",)

    ctx = scallopy.ScallopContext()

    # Register the semantic parser
    ctx.register_foreign_predicate(my_dummy_semantic_parser)

    # Add a program
    ctx.add_program("""
      type Expr = Const(i32) | Add(Expr, Expr)
      rel eval(e, v)       = case e is Const(v)
      rel eval(e, v1 + v2) = case e is Add(e1, e2) and eval(e1, v1) and eval(e2, v2)
      rel prompt = {"If I have 3 apples and 2 pears, how many fruits do I have?"}
      rel result(v) = prompt(p) and my_dummy_semantic_parser(p, e) and eval(e, v)
    """)

    # Run the context
    ctx.run()

    # The result should be 5
    self.assertEqual(list(ctx.relation("result")), [(5,)])

  def test_fp_suppress_warning(self):
    @scallopy.foreign_predicate(suppress_warning=True)
    def dummy(s: str) -> scallopy.Generator[None, str]:
      raise Exception("always false")

    ctx = scallopy.ScallopContext()
    ctx.register_foreign_predicate(dummy)
    ctx.add_program("""rel result(y) = dummy("hello", y)""")
    ctx.run()
    self.assertEqual(list(ctx.relation("result")), [])
