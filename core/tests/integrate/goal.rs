use scallop_core::{runtime::env::RuntimeEnvironmentOptions, testing::*};

#[test]
fn simple_edge_path_with_goal() {
  expect_interpret_result_with_runtime_option(
    r#"
      rel edge = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)}
      @demand("bf")
      rel path(a, b) = edge(a, b) \/ path(a, c) /\ edge(c, b)
      @goal
      rel goal() = path(0, 2)
    "#,
    RuntimeEnvironmentOptions::default().with_stop_when_goal_non_empty(true),
    ("path", vec![(0, 1), (0, 2)]),
  );
}

#[test]
fn simple_edge_path_with_disjunctive_goal() {
  expect_interpret_result_with_runtime_option(
    r#"
      rel edge = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)}
      @demand("bf")
      rel path(a, b) = edge(a, b) \/ path(a, c) /\ edge(c, b)
      @goal
      rel goal() = path(0, 2) or path(0, 5)
    "#,
    RuntimeEnvironmentOptions::default().with_stop_when_goal_non_empty(true),
    ("path", vec![(0, 1), (0, 2)]),
  );
}

#[test]
fn simple_edge_path_with_multiple_goals() {
  expect_front_compile_failure(
    r#"
      rel edge = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)}
      @demand("bf")
      rel path(a, b) = edge(a, b) \/ path(a, c) /\ edge(c, b)
      @goal rel goal1() = path(0, 2)
      @goal rel goal2() = path(0, 1)
    "#,
    |err| err.contains("There are more than one relations that are annotated with @goal"),
  );
}

#[test]
fn simple_edge_path_with_non_nullary_goal() {
  expect_front_compile_failure(
    r#"
      rel edge = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)}
      @demand("bf")
      rel path(a, b) = edge(a, b) \/ path(a, c) /\ edge(c, b)
      @goal rel goal(x) = path(0, x)
    "#,
    |err| err.contains("@goal annotated relation must be of arity-0"),
  );
}
