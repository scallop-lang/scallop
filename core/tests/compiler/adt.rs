use scallop_core::integrate::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;
use scallop_core::utils::*;

#[test]
fn adt_duplicated_name_1() {
  expect_front_compile_failure(
    r#"
      type Expr = Const(i32) | Add(Expr, Expr) | Minus(Expr, Expr) | Add(Expr, Expr)
    "#,
    |e| e.contains("duplicate"),
  )
}

#[test]
fn adt_duplicated_name_2() {
  expect_front_compile_failure(
    r#"
      type Expr = Const(i32) | Add(Expr, Expr)
      type Expr2 = Const(i32) | Sub(Expr, Expr)
    "#,
    |e| e.contains("duplicate"),
  )
}

#[test]
fn adt_entity() {
  expect_compile(
    r#"
      type Expr = Const(i32) | Add(Expr, Expr)
      const MY_EXPR = Add(Const(5), Const(3))
    "#,
  )
}

#[test]
fn adt_entity_fail_1() {
  expect_front_compile_failure(
    r#"
      type Expr = Const(i32) | Add(Expr, Expr)
      const MY_EXPR = Add(Const(5 + 5), Const(3))
    "#,
    |e| e.contains("non-constant"),
  )
}

#[test]
fn adt_entity_fail_2() {
  expect_front_compile_failure(
    r#"
      type Expr = Const(i32) | Add(Expr, Expr)
      const MY_EXPR = Sub(Const(5), Const(3))
    "#,
    |e| e.contains("unknown algebraic data type variant"),
  )
}

#[test]
fn adt_entity_arity_mismatch_1() {
  expect_front_compile_failure(
    r#"
      type Expr = Const(i32) | Add(Expr, Expr)
      const MY_EXPR = Add(Const(5), Const(3), Const(7))
    "#,
    |e| e.contains("arity mismatch"),
  )
}

#[test]
fn adt_entity_type_error_1() {
  expect_front_compile_failure(
    r#"
      type Expr = Const(i32) | Add(Expr, Expr)
      const MY_EXPR = Add(Const(5), Const("this is a string"))
    "#,
    |e| e.contains("cannot unify types"),
  )
}

#[test]
fn adt_add_dynamic_entity_1() {
  let prov = unit::UnitProvenance::new();
  let mut ctx = IntegrateContext::<unit::UnitProvenance, RcFamily>::new(prov);

  // Compile a program containing ADT definitions
  ctx
    .add_program(
      r#"
      type Expr = Const(i32) | Add(Expr, Expr)
      type root(e: Expr)
      rel eval(e, y) = case e is Const(y)
      rel eval(e, y1 + y2) = case e is Add(e1, e2) and eval(e1, y1) and eval(e2, y2)
      rel result(y) = root(e) and eval(e, y)
    "#,
    )
    .expect("Compile error");

  // Dynamically add an entity to the context
  ctx
    .add_entity("root", vec!["Add(Const(5), Add(Const(2), Const(3)))".to_string()])
    .expect("Cannot add entity");

  // Run the context
  ctx.run().expect("Runtime error");

  // Check the results
  expect_output_collection(
    "result",
    ctx.computed_relation_ref("result").expect("Cannot get result"),
    vec![(10i32,)],
  );
}

#[test]
fn adt_add_dynamic_entity_2() {
  let prov = unit::UnitProvenance::new();
  let mut ctx = IntegrateContext::<unit::UnitProvenance, RcFamily>::new(prov);

  // Compile a program containing ADT definitions
  ctx
    .add_program(
      r#"
      type Expr = Const(i32) | Add(Expr, Expr)
      type root(id: i32, e: Expr)
      rel eval(e, y) = case e is Const(y)
      rel eval(e, y1 + y2) = case e is Add(e1, e2) and eval(e1, y1) and eval(e2, y2)
      rel result(id, y) = root(id, e) and eval(e, y)
    "#,
    )
    .expect("Compile error");

  // Dynamically add an entity to the context
  ctx
    .add_entity(
      "root",
      vec!["1".to_string(), "Add(Const(5), Add(Const(2), Const(3)))".to_string()],
    )
    .expect("Cannot add entity");
  ctx
    .add_entity("root", vec!["2".to_string(), "Const(3)".to_string()])
    .expect("Cannot add entity");

  // Run the context
  ctx.run().expect("Runtime error");

  // Check the results
  expect_output_collection(
    "result",
    ctx.computed_relation_ref("result").expect("Cannot get result"),
    vec![(1i32, 10i32), (2, 3)],
  );
}
