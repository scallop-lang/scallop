use scallop_core::integrate;
use scallop_core::runtime::provenance;
use scallop_core::testing::*;
use scallop_core::utils::RcFamily;

#[test]
fn incr_edge_path_left_recursion() {
  let prov_ctx = provenance::unit::UnitProvenance::default();
  let mut ctx = integrate::IntegrateContext::<_, RcFamily>::new(prov_ctx);

  // Source
  ctx.add_relation("edge(usize, usize)").unwrap();
  ctx
    .add_rule(r#"path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)"#)
    .unwrap();

  // Facts
  ctx
    .add_facts(
      "edge",
      vec![(None, (0usize, 1usize).into()), (None, (1usize, 2usize).into())],
      false,
    )
    .unwrap();

  // Execution
  ctx.run().unwrap();

  // Result
  expect_output_collection(
    ctx.computed_relation("path").unwrap(),
    vec![(0usize, 1usize), (0, 2), (1, 2)],
  );
}

#[test]
fn incr_edge_path_left_branching_1() {
  let prov_ctx = provenance::unit::UnitProvenance::default();
  let mut ctx = integrate::IntegrateContext::<_, RcFamily>::new(prov_ctx);

  // Base context
  ctx.add_relation("edge(usize, usize)").unwrap();
  ctx
    .add_facts(
      "edge",
      vec![(None, (0usize, 1usize).into()), (None, (1usize, 2usize).into())],
      false,
    )
    .unwrap();
  ctx.run().unwrap();
  expect_output_collection(ctx.computed_relation("edge").unwrap(), vec![(0usize, 1usize), (1, 2)]);

  // First branch
  let mut first_branch = ctx.clone();
  first_branch
    .add_rule(r#"path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)"#)
    .unwrap();
  first_branch.run().unwrap();
  expect_output_collection(
    first_branch.computed_relation("path").unwrap(),
    vec![(0usize, 1usize), (0, 2), (1, 2)],
  );

  // Second branch
  let mut second_branch = ctx.clone();
  second_branch
    .add_rule(r#"path(a, c) = edge(a, c) \/ edge(a, b) /\ path(b, c)"#)
    .unwrap();
  second_branch.run().unwrap();
  expect_output_collection(
    second_branch.computed_relation("path").unwrap(),
    vec![(0usize, 1usize), (0, 2), (1, 2)],
  );

  // Second branch, continuation
  second_branch.add_rule(r#"result(1, y) = path(1, y)"#).unwrap();
  second_branch.run().unwrap();
  expect_output_collection(
    second_branch.computed_relation("result").unwrap(),
    vec![(1usize, 2usize)],
  );
}
