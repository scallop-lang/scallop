use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn test_how_many_3_add_mult() {
  let mut ctx = add_mult_prob::AddMultProbContext::default();
  expect_interpret_result_with_tag(
    r#"
      rel digit = {0.91::(0, 0), 0.01::(0, 1), 0.01::(0, 2), 0.01::(0, 3)}
      rel result(n) :- n = count(o: digit(o, 3))
    "#,
    &mut ctx,
    ("result", vec![(0.99, (0usize,)), (0.01, (1usize,))]),
    add_mult_prob::AddMultProbContext::soft_cmp,
  )
}

#[test]
fn test_min_max_with_recursion() {
  let mut ctx = min_max_prob::MinMaxProbProvenance::default();
  expect_interpret_result_with_tag(
    r#"
      rel enemy = {
        0.1::(1, 3), 0.1::(2, 3), 0.1::(3, 3),
        0.1::(1, 2), 0.8::(2, 2), 0.9::(3, 2),
        0.1::(1, 1), 0.1::(2, 1), 0.1::(3, 1),
      }
      rel edge = {
        (1, 1, 1, 2), (1, 1, 2, 1),
        (1, 2, 1, 3), (1, 2, 2, 2), (1, 2, 1, 1),
        (1, 3, 1, 2), (1, 3, 2, 3),
        (2, 1, 1, 1), (2, 1, 2, 2), (2, 1, 3, 1),
        (2, 2, 1, 2), (2, 2, 2, 3), (2, 2, 2, 1), (2, 2, 3, 2),
        (2, 3, 1, 3), (2, 3, 2, 2), (2, 3, 3, 3),
        (3, 1, 2, 1), (3, 1, 3, 2),
        (3, 2, 2, 2), (3, 2, 3, 1), (3, 2, 3, 3),
        (3, 3, 2, 3), (3, 3, 3, 2),
      }
      rel path(xa, ya, xb, yb) = edge(xa, ya, xb, yb), not enemy(xb, yb)
      rel path(xa, ya, xc, yc) = path(xa, ya, xb, yb), edge(xb, yb, xc, yc), not enemy(xc, yc)
      query path(3, 1, 3, 3)
    "#,
    &mut ctx,
    ("path(3, 1, 3, 3)", vec![(0.9, (3usize, 1usize, 3usize, 3usize))]),
    min_max_prob::MinMaxProbProvenance::cmp,
  )
}
