use scallop_core::testing::*;

#[test]
fn multi_character_literal() {
  expect_front_compile_failure(
    r#"
    rel bad = {'asdf'}
    "#,
    |e| e.contains("invalid character"),
  )
}

#[test]
fn var_in_fact_decl_1() {
  expect_front_compile_failure(
    r#"
    rel bad(x, x + 1)
    "#,
    |e| e.contains("unknown variable `x`"),
  )
}

#[test]
fn unknown_function_1() {
  expect_front_compile_failure(
    r#"
    rel bad($asdf(1, 3))
    "#,
    |e| e.contains("unknown function `asdf`"),
  )
}

#[test]
fn abs_type_mismatch_1() {
  expect_front_compile_failure(
    r#"
    type A(usize)
    type B(i32)
    rel B($abs(x)) :- A(x)
    "#,
    |e| e.contains("cannot unify type"),
  )
}

#[test]
fn cannot_cast_type_1() {
  expect_front_compile_failure(
    r#"
    type A(f32)
    type B(char)
    rel B(x as char) = A(x)
    "#,
    |e| e.contains("cannot cast type from `f32` to `char`"),
  )
}

#[test]
fn duplicated_relation_decl_1() {
  expect_front_compile_failure(
    r#"
    type A(f32)
    type A(char)
    "#,
    |e| e.contains("duplicated relation type declaration"),
  )
}

#[test]
fn duplicated_relation_decl_2() {
  expect_front_compile_failure(
    r#"
    type A(f32)
    type A(f32)
    "#,
    |e| e.contains("duplicated relation type declaration"),
  )
}

#[test]
fn duplicated_constant_decl_1() {
  expect_front_compile_failure(
    r#"
    const ABC = 3
    const ABC = 4
    "#,
    |e| e.contains("duplicated declaration of constant `ABC`"),
  )
}

#[test]
fn conflicting_constant_decl_type_1() {
  expect_front_compile_failure(
    r#"
    const ABC: String = 3
    "#,
    |e| e.contains("cannot unify"),
  )
}

#[test]
fn conflicting_constant_decl_type_2() {
  expect_front_compile_failure(
    r#"
    const ABC: usize = -5
    "#,
    |e| e.contains("cannot unify"),
  )
}

#[test]
fn conflicting_constant_decl_type_3() {
  expect_front_compile_failure(
    r#"
    const V: usize = 5
    type r(u32, usize)
    rel r = {(V, 3), (3, V)}
    "#,
    |e| e.contains("cannot unify"),
  )
}

#[test]
fn bad_enum_type_decl() {
  expect_front_compile_failure(
    r#"
    type K = A = 3 | B | C = 4 | D
    "#,
    |e| e.contains("has already been assigned"),
  )
}
