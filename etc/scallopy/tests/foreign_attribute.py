import unittest

import scallopy

@scallopy.foreign_attribute
def add_something(item, v1):
  # Check if the annotation is on function type decl
  assert item.is_function_decl(), "[@add_something] has to be an attribute of a function type declaration"

  # Check the input and return types
  arg_types = item.function_decl_arg_types()
  assert len(arg_types) == 1 and arg_types[0].is_i32(), "[@add_something] expects a function (i32) -> i32"
  assert item.function_decl_ret_type().is_i32(), "[@add_something] expects a function (i32) -> i32"

  # Define the foreign function
  @scallopy.foreign_function(name=item.function_decl_name())
  def add(v2: scallopy.i32) -> scallopy.i32: return v1 + v2

  # Tell compiler to remove the item and add a foreign function
  return add


@scallopy.foreign_attribute
def string_join(item, separator=","):
  # Needs to be a relation declaration, and can have only one relation
  assert item.is_relation_decl(), "[@string_join] has to be an attribute of a relation type declaration"
  assert len(item.relation_decls()) == 1, "[@string_join] can annotate only one relation type declaration"
  relation_decl = item.relation_decl(0)

  # Need to have more than one argument and all need to be string
  arg_types = [ab.ty for ab in relation_decl.arg_bindings]
  assert len(arg_types) > 1, "[@string_join] has to have more than 1 argument"
  assert all([at.is_string() for at in arg_types]), "[@string_join] all arguments need to have `String` type"

  # Adornments should be `bound*free`
  arg_adornments = [ab.adornment for ab in relation_decl.arg_bindings]
  assert all([a.is_bound() for a in arg_adornments[:-1]]), "[@string_join] all argument except the last one should be bound"
  assert arg_adornments[-1] is None or arg_adornments[-1].is_free(), "[@string_join] last argument should be free"

  # Generate the foreign predicate
  @scallopy.foreign_predicate(name=relation_decl.name.name, input_arg_types=arg_types[:-1], output_arg_types=arg_types[-1:])
  def f(*args) -> scallopy.Facts[None, str]:
    yield (separator.join(args),)

  # Return the foreign predicate to indicate that we need to remove the item and add a foreign predicate
  return f


class TestForeignAttribute(unittest.TestCase):
  def test_register_function_1(self):
    ctx = scallopy.ScallopContext()
    ctx.register_foreign_attribute(add_something)
    ctx.add_program("""
      @add_something(3)
      type $compute(x: i32) -> i32
      rel S = {(1,)}
      rel R($compute(x)) = S(x)
    """)
    ctx.run()
    self.assertEqual(list(ctx.relation("R")), [(4,)])

  @unittest.expectedFailure
  def test_register_function_type_check_fail_1(self):
    """
    Since the type definition is $compute(x: usize) -> i32, the type
    usize does not match the expected `i32`, thus fail
    """
    ctx = scallopy.ScallopContext()
    ctx.register_foreign_attribute(add_something)
    ctx.add_program("""
      @add_something(3)
      type $compute(x: usize) -> i32
      rel S = {(1,)}
      rel R($compute(x)) = S(x)
    """)
    ctx.run()
    self.assertEqual(list(ctx.relation("R")), [(4,)])

  @unittest.expectedFailure
  def test_register_function_fail_2(self):
    """
    Since the type definition is $compute(x: usize) -> i32, the type
    usize does not match the expected `i32`, thus fail
    """
    ctx = scallopy.ScallopContext()
    ctx.register_foreign_attribute(add_something)
    ctx.add_program("""
      @add_something
      type $compute(x: usize) -> i32
      rel S = {(1,)}
      rel R($compute(x)) = S(x)
    """)

  def test_register_predicate_1(self):
    ctx = scallopy.ScallopContext()
    ctx.register_foreign_attribute(string_join)
    ctx.add_program("""
      @string_join
      type join3(bound x: String, bound y: String, bound z: String, w: String)
      rel S = {("x", "y", "z")}
      rel R(w) = S(x, y, z) and join3(x, y, z, w)
    """)
    ctx.run()
    self.assertEqual(list(ctx.relation("R")), [("x,y,z",)])

  def test_register_predicate_2(self):
    ctx = scallopy.ScallopContext()
    ctx.register_foreign_attribute(string_join)
    ctx.add_program("""
      @string_join("-") type join3(bound x: String, bound y: String, bound z: String, w: String)
      rel S = {("x", "y", "z")}
      rel R(w) = S(x, y, z) and join3(x, y, z, w)
    """)
    ctx.run()
    self.assertEqual(list(ctx.relation("R")), [("x-y-z",)])

  def test_register_predicate_3(self):
    ctx = scallopy.ScallopContext()
    ctx.register_foreign_attribute(string_join)
    ctx.add_program("""
      @string_join("-") type join3(bound x: String, bound y: String, bound z: String, free w: String)
      rel S = {("x", "y", "z")}
      rel R(w) = S(x, y, z) and join3(x, y, z, w)
    """)
    ctx.run()
    self.assertEqual(list(ctx.relation("R")), [("x-y-z",)])

  @unittest.expectedFailure
  def test_register_predicate_fail_1(self):
    ctx = scallopy.ScallopContext()
    ctx.register_foreign_attribute(string_join)
    ctx.add_program("""
      @string_join("-") type join3(bound x: String, bound y: String, z: String, w: String)
      rel S = {("x", "y", "z")}
      rel R(w) = S(x, y, z) and join3(x, y, z, w)
    """)

  @unittest.expectedFailure
  def test_register_predicate_fail_2(self):
    ctx = scallopy.ScallopContext()
    ctx.register_foreign_attribute(string_join)
    ctx.add_program("""
      @string_join("-") type join3(bound x: String, bound y: String, bound z: String, bound w: String)
      rel S = {("x", "y", "z")}
      rel R(w) = S(x, y, z) and join3(x, y, z, w)
    """)
