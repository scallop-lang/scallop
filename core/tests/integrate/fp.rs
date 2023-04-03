use scallop_core::testing::*;

#[test]
fn range_free_1() {
  expect_interpret_result(
    r#"
      rel result(y) = range_usize(0, 5, y)
    "#,
    ("result", vec![(0usize,), (1,), (2,), (3,), (4,)]),
  );
}

#[test]
fn range_constraint_1() {
  expect_interpret_result(
    r#"
      rel base = {("A", "B", 3.0)}
      rel result(a, b) = base(a, b, x) and soft_eq_f32(x, 3.0)
    "#,
    ("result", vec![("A".to_string(), "B".to_string())]),
  );
}

#[test]
fn range_join_1() {
  expect_interpret_result(
    r#"
      rel base = {3}
      rel result(y) = base(x) and range_usize(0, x, y)
    "#,
    ("result", vec![(0usize,), (1,), (2,)]),
  );
}

#[test]
fn range_join_2() {
  expect_interpret_result(
    r#"
      rel base = {3}
      rel result() = base(x) and range_usize(0, x, 2)
    "#,
    ("result", vec![()]),
  );
}

#[test]
fn range_join_3() {
  expect_interpret_empty_result(
    r#"
      rel base = {3}
      rel result() = base(x) and range_usize(0, x, 100)
    "#,
    "result",
  );
}

#[test]
fn range_join_4() {
  expect_interpret_result(
    r#"
      rel base = {3, 10}
      rel result(x) = base(x) and range_usize(0, x, 5)
    "#,
    ("result", vec![(10usize,)]),
  );
}

#[test]
fn string_chars_1() {
  expect_interpret_result(
    r#"
      rel string = {"hello"}
      rel result(i, c) = string(s), string_chars(s, i, c)
    "#,
    ("result", vec![(0usize, 'h'), (1, 'e'), (2, 'l'), (3, 'l'), (4, 'o')]),
  );
}

#[test]
fn floating_point_eq_1() {
  expect_interpret_multi_result(
    r#"
      rel result_1() = float_eq_f32(3.000001, 1.000001 + 2.000001)
      rel result_2() = 3.000001 == 1.000001 + 2.000001
    "#,
    vec![
      ("result_1", vec![()].into()),
      ("result_2", TestCollection::empty())
    ],
  )
}
