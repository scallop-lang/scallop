use scallop_core::runtime::provenance::add_mult_prob;
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
