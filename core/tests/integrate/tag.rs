use scallop_core::runtime::provenance;
use scallop_core::testing::*;

#[test]
fn test_rule_constant_tag_simple_1() {
  use provenance::add_mult_prob::*;
  let prov_ctx = AddMultProbProvenance::default();
  expect_interpret_result_with_tag(
    r#"
      rel my_num(5)
      rel 0.5::fall_off(n) = my_num(n)
    "#,
    prov_ctx,
    ("fall_off", vec![(0.5, (5i32,))]),
    AddMultProbProvenance::soft_cmp,
  )
}

#[test]
fn test_rule_constant_integer_tag_simple_1() {
  use provenance::add_mult_prob::*;
  let prov_ctx = AddMultProbProvenance::default();
  expect_interpret_result_with_tag(
    r#"
      rel my_num(5)
      rel 1::fall_off(n) = my_num(n)
    "#,
    prov_ctx,
    ("fall_off", vec![(1.0, (5i32,))]),
    AddMultProbProvenance::soft_cmp,
  )
}

#[test]
fn test_multiple_rule_constant_tag_simple_1() {
  use provenance::add_mult_prob::*;
  let prov_ctx = AddMultProbProvenance::default();
  expect_interpret_result_with_tag(
    r#"
      rel my_num(5)
      rel 0.5::fall_off(n / 2) = my_num(n)
      rel 0.2::fall_off(n + 1) = my_num(n)
    "#,
    prov_ctx,
    ("fall_off", vec![(0.5, (2i32,)), (0.2, (6i32,))]),
    AddMultProbProvenance::soft_cmp,
  )
}

#[test]
fn test_expr_tag_direct_propagate_1() {
  use provenance::add_mult_prob::*;
  let prov_ctx = AddMultProbProvenance::default();
  expect_interpret_result_with_tag(
    r#"
      rel my_prob(0.5)
      rel p::fall_off() = my_prob(p)
    "#,
    prov_ctx,
    ("fall_off", vec![(0.5, ())]),
    AddMultProbProvenance::soft_cmp,
  )
}

#[test]
fn test_expr_tag_simple_1() {
  use provenance::add_mult_prob::*;
  let prov_ctx = AddMultProbProvenance::default();
  expect_interpret_result_with_tag(
    r#"
      rel my_num(5.0)
      rel 1.0/n::fall_off() = my_num(n)
    "#,
    prov_ctx,
    ("fall_off", vec![(0.2, ())]),
    AddMultProbProvenance::soft_cmp,
  )
}

#[test]
fn test_expr_tag_simple_2() {
  use provenance::min_max_prob::*;
  let prov_ctx = MinMaxProbProvenance::default();
  expect_interpret_result_with_tag(
    r#"
      rel edge = {(0, 1), (1, 2)}
      rel path(x, y, 1.0) = edge(x, y)
      rel path(x, z, l + 1.0) = path(x, y, l) and edge(y, z)
      rel (1.0 / l)::path_prob(x, y) = path(x, y, l)
    "#,
    prov_ctx,
    ("path_prob", vec![(1.0, (0, 1)), (1.0, (1, 2)), (0.5, (0, 2))]),
    MinMaxProbProvenance::cmp,
  )
}

#[test]
fn test_expr_tag_type_error() {
  // `1.0 / n` will fail because `n` is of type `i32`
  expect_compile_failure(
    r#"
      type my_num(i32)
      rel my_num(5)
      rel (1.0 / n)::fall_off() = my_num(n)
    "#,
    |err| err.to_string().contains("type"),
  )
}

#[test]
fn test_expr_tag_unbound_error() {
  expect_compile_failure(
    r#"
      rel my_num()
      rel (1.0 / n)::fall_off() = my_num()
    "#,
    |err| err.to_string().contains("bound"),
  )
}

#[test]
fn test_expr_tag_cannot_be_datetime() {
  expect_compile_failure(
    r#"
      rel my_time(t"2024-01-01")
      rel t::fall_off() = my_time(t)
    "#,
    |err| {
      err
        .to_string()
        .contains("A value of type `DateTime` cannot be casted into a dynamic tag")
    },
  )
}

#[test]
fn test_tropical_min_path_length() {
  const A: usize = 0;
  const B: usize = 1;
  const C: usize = 2;
  const D: usize = 3;

  use provenance::tropical::*;
  let prov_ctx = TropicalSemiring::default();
  expect_interpret_result_with_tag(
    r#"
      type Node = A | B | C | D
      rel edge = {
        1::(A, B), 1::(B, C), 1::(C, D),
        1::(A, D),
      }
      rel path(x, y) = edge(x, y) or path(x, z) and edge(z, y)
      query path
    "#,
    prov_ctx,
    (
      "path",
      vec![
        (1, (A, B)),
        (1, (B, C)),
        (1, (C, D)),
        (2, (A, C)),
        (2, (B, D)),
        (1, (A, D)),
      ],
    ),
    usize::eq,
  )
}
