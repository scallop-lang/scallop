use scallop_core::integrate::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;
use scallop_core::utils::*;

#[test]
fn edb_edge_path_left_recursion() {
  expect_interpret_result_with_setup(
    r#"
      type edge(usize, usize)
      rel path(a, b) = edge(a, b) \/ path(a, c) /\ edge(c, b)
      query path
    "#,
    |edb| {
      edb
        .add_facts("edge", vec![(0usize, 2usize), (1, 2), (2, 3)])
        .expect("Error adding facts");
    },
    ("path", vec![(0usize, 2usize), (1, 2), (0, 3), (1, 3), (2, 3)]),
  );
}

#[test]
fn edb_edge_path_right_recursion() {
  expect_interpret_result_with_setup(
    r#"
      type edge(usize, usize)
      rel path(a, b) = edge(a, b) or (edge(a, c) and path(c, b))
      query path
    "#,
    |edb| {
      edb
        .add_facts("edge", vec![(0usize, 2usize), (1, 2), (2, 3)])
        .expect("Error adding facts");
    },
    ("path", vec![(0usize, 2usize), (1, 2), (0, 3), (1, 3), (2, 3)]),
  );
}

#[test]
fn edb_edge_path_both_program_and_external() {
  expect_interpret_result_with_setup(
    r#"
      type edge(usize, usize)
      rel edge = {(2, 3)}
      rel path(a, b) = edge(a, b) or (edge(a, c) and path(c, b))
      query path
    "#,
    |edb| {
      edb
        .add_facts("edge", vec![(0usize, 2usize), (1, 2)])
        .expect("Error adding facts");
    },
    ("path", vec![(0usize, 2usize), (1, 2), (0, 3), (1, 3), (2, 3)]),
  );
}

#[test]
fn edb_cannot_incremental_update_program_facts() {
  let prov = unit::UnitProvenance::default();
  let mut ctx = IntegrateContext::<_, RcFamily>::new(prov);

  // First interpret a program with edge facts declared in program
  ctx
    .add_program(
      r#"
      rel edge = {(0, 1), (1, 2)}
      rel path(a, c) = edge(a, c) or (path(a, b) and edge(b, c))
      query path
    "#,
    )
    .expect("Compilation error");

  // Execute it
  ctx.run().expect("Runtime error");

  // Compile some new things into the context
  ctx
    .add_program(
      r#"
      rel edge = {(2, 3)}
    "#,
    )
    .expect("Compilation error");

  // When executing, it should throw error
  ctx.run().expect_err("Database Error");
}

#[test]
fn edb_edge_path_incremental_update() {
  let prov = unit::UnitProvenance::default();
  let mut ctx = IntegrateContext::<_, RcFamily>::new(prov);

  // First interpret a program with edge facts declared in program
  ctx
    .add_program(
      r#"
      type edge(usize, usize)
      rel path(a, c) = edge(a, c) or (path(a, b) and edge(b, c))
      query path
    "#,
    )
    .expect("Compilation error");

  ctx
    .edb()
    .add_facts("edge", vec![(0usize, 1usize), (1, 2)])
    .expect("Cannot add facts");

  // Execute it
  ctx.run().expect("Runtime error");

  // Check the result
  expect_output_collection(
    "path",
    ctx.computed_relation_ref("path").unwrap(),
    vec![(0usize, 1usize), (0, 2), (1, 2)],
  );

  // Compile some new things into the context
  ctx
    .edb()
    .add_facts("edge", vec![(2usize, 3usize)])
    .expect("Cannot add facts");

  // When executing, it should throw error
  ctx.run().expect("Runtime error");

  // Check the result
  expect_output_collection(
    "path",
    ctx.computed_relation_ref("path").unwrap(),
    vec![(0usize, 1usize), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
  );
}

#[test]
fn edb_fib_1() {
  expect_interpret_result_with_setup(
    r#"
      type fib(i32, usize)
      rel fib(x, a + b) = fib(x - 1, a), fib(x - 2, b), x <= 5
    "#,
    |edb| {
      edb
        .add_facts("fib", vec![(0i32, 1usize), (1, 1)])
        .expect("Error adding facts");
    },
    ("fib", vec![(0i32, 1usize), (1, 1), (2, 2), (3, 3), (4, 5), (5, 8)]),
  );
}

#[test]
fn edb_edge_path_persistent_relation() {
  let prov = unit::UnitProvenance::default();
  let mut ctx = IntegrateContext::<_, RcFamily>::new(prov);

  // First interpret a program with edge facts declared in program
  ctx
    .add_program(
      r#"
      type edge_1(usize, usize), edge_2(usize, usize)
      rel path_1(a, c) = edge_1(a, c) or (path_1(a, b) and edge_1(b, c))
      rel path_2(a, c) = edge_2(a, c) or (path_2(a, b) and edge_2(b, c))
      query path_1
      query path_2
    "#,
    )
    .expect("Compilation error");

  ctx
    .edb()
    .add_facts("edge_1", vec![(0usize, 1usize), (1, 2)])
    .expect("Cannot add facts");
  ctx
    .edb()
    .add_facts("edge_2", vec![(0usize, 1usize), (1, 2)])
    .expect("Cannot add facts");

  // Assert need update relations
  let need_update_relations = ctx.internal_context().exec_ctx.edb.need_update_relations();
  assert!(need_update_relations.is_empty());

  // Execute it
  ctx.run().expect("Runtime error");

  // Assert need update relations
  let need_update_relations = ctx.internal_context().exec_ctx.edb.need_update_relations();
  assert!(need_update_relations.is_empty());

  // Check the result
  expect_output_collection(
    "path_1",
    ctx.computed_relation_ref("path_1").unwrap(),
    vec![(0usize, 1usize), (0, 2), (1, 2)],
  );
  expect_output_collection(
    "path_2",
    ctx.computed_relation_ref("path_2").unwrap(),
    vec![(0usize, 1usize), (0, 2), (1, 2)],
  );

  // Compile some new things into the context
  ctx
    .edb()
    .add_facts("edge_2", vec![(2usize, 3usize)])
    .expect("Cannot add facts");

  // Assert the need update
  let need_update_relations = ctx.internal_context().exec_ctx.edb.need_update_relations();
  assert!(need_update_relations.contains("edge_2"));
  assert!(!need_update_relations.contains("edge_1"));

  // When executing, it should throw error
  ctx.run().expect("Runtime error");

  // Check the result
  expect_output_collection(
    "path_1",
    ctx.computed_relation_ref("path_1").unwrap(),
    vec![(0usize, 1usize), (0, 2), (1, 2)],
  );
  expect_output_collection(
    "path_2",
    ctx.computed_relation_ref("path_2").unwrap(),
    vec![(0usize, 1usize), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
  );
}
