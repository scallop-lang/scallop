use scallop_core::testing::*;

#[test]
fn dt_fib_1() {
  expect_interpret_result(
    r#"
      rel fib = {(0, 1), (1, 1)}
      @demand("bf")
      rel fib(x, a + b) = fib(x - 1, a), fib(x - 2, b), x > 1
      query fib(10, y)
    "#,
    ("fib(10, y)", vec![(10i32, 89i32)]),
  );
}

#[test]
fn dt_range_1() {
  expect_interpret_result(
    r#"
      @demand("bbf")
      rel range(a, b, i) = a == i
      rel range(a, b, i) = range(a, b, i - 1), i < b
      query range(1, 4, x)
    "#,
    ("range(1, 4, x)", vec![(1i32, 4i32, 1i32), (1, 4, 2), (1, 4, 3)]),
  );
}

#[test]
fn dt_edge_path_1() {
  expect_interpret_result(
    r#"
      rel edge = {(0, 1), (1, 2), (2, 3)}
      @demand("fb")
      rel path(a, b) = edge(a, b) \/ path(a, c) /\ edge(c, b)
      query path(_, 3)
    "#,
    ("path(_, 3)", vec![(0, 3), (1, 3), (2, 3)]),
  );
}

#[test]
fn dt_edge_path_2() {
  expect_interpret_result(
    r#"
      rel edge = {(0, 1), (1, 2), (2, 3)}
      @demand("bf")
      rel path(a, b) = edge(a, b) \/ path(a, c) /\ edge(c, b)
      query path(0, _)
    "#,
    ("path(0, _)", vec![(0, 1), (0, 2), (0, 3)]),
  );
}
