import unittest

import scallopy

class TestForeignFunction(unittest.TestCase):
  def test_foreign_sum(self):
    """
    This function tests variable argument foreign function with generic type parameter
    """

    # First create the foreign sum function
    T = scallopy.GenericTypeParameter(scallopy.Number)
    @scallopy.foreign_function
    def my_sum(*numbers: T) -> T:
      s = 0
      for x in numbers:
        s += x
      return s

    # Then add the context
    ctx = scallopy.ScallopContext()
    ctx.register_foreign_function(my_sum)
    ctx.add_relation("S", (scallopy.usize, scallopy.usize))
    ctx.add_facts("S", [(1, 2), (2, 3)])
    ctx.add_relation("T", (scallopy.i8, scallopy.i8, scallopy.i8))
    ctx.add_facts("T", [(1, 2, 3), (5, 6, 7)])
    ctx.add_rule("R($my_sum(a, b) as i32) = S(a, b)")
    ctx.add_rule("R($my_sum(a, b, c) as i32) = T(a, b, c)")
    ctx.run()
    self.assertEqual(list(ctx.relation("R")), [(3,), (5,), (6,), (18,)])

  def test_foreign_string_index_of(self):
    """
    This function tests string_index_of which could produce runtime errors
    """

    # First create the foreign function
    @scallopy.foreign_function
    def my_string_index_of(s1: str, s2: str) -> scallopy.usize:
      return s1.index(s2)

    # Then add the context
    ctx = scallopy.ScallopContext()
    ctx.register_foreign_function(my_string_index_of)
    ctx.add_relation("S", (str, str))
    ctx.add_facts("S", [("hello world", "hello"), ("hello world", "world"), ("hello world", "42")])
    ctx.add_rule("R($my_string_index_of(a, b)) = S(a, b)")
    ctx.run()
    self.assertEqual(list(ctx.relation("R")), [(0,), (6,)])

  @unittest.expectedFailure
  def test_register_non_foreign_function(self):
    def my_sum(a, b): return a + b
    ctx = scallopy.Context()
    ctx.register_foreign_function(my_sum)

  @unittest.expectedFailure
  def test_bad_foreign_function_1(self):
    """
    No type annotation
    """
    @scallopy.ff
    def my_sum(a, b) -> scallopy.usize:
      return a + b

  @unittest.expectedFailure
  def test_bad_foreign_function_2(self):
    """
    Return type generic but not bounded
    """
    T = scallopy.Generic()
    @scallopy.ff
    def my_sum(a: scallopy.usize, b: scallopy.usize) -> T:
      return a + b

  @unittest.expectedFailure
  def test_bad_foreign_function_3(self):
    """
    Return type is a type family
    """
    @scallopy.ff
    def my_sum(a: scallopy.usize, b: scallopy.usize) -> int:
      return a + b

  @unittest.expectedFailure
  def test_add_foreign_function_twice(self):
    """
    Return type generic but not bounded
    """
    @scallopy.ff
    def my_sum(a: scallopy.usize, b: scallopy.usize) -> scallopy.usize:
      return a + b
    ctx = scallopy.Context()
    ctx.register_foreign_function(my_sum)
    ctx.register_foreign_function(my_sum) # register it twice, should fail
