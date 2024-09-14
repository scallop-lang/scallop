use scallop_core::{
  runtime::{
    env::RuntimeEnvironmentOptions,
    env::Scheduler,
    provenance,
  },
  testing::*,
  utils::RcFamily,
};

#[test]
fn prob_edge_path_with_goal_and_a_star_search() {
  expect_interpret_within_iter_limit_with_ctx_and_runtime_options::<
    provenance::min_max_prob::MinMaxProbProvenance,
    RcFamily,
  >(
    r#"
      rel edge = {
        0.2::(0, 100),
        0.2::(100, 101),
        0.2::(101, 1000),

        0.99::(0, 200),
        0.99::(200, 201),
        0.99::(201, 202),
        0.99::(202, 203),
        0.99::(203, 1000),

        0.98::(1000, 2000),
        0.97::(2000, 3000),
        0.96::(3000, 4000),
      }

      @demand("bf")
      rel path(a, b) = edge(a, b) \/ path(a, c) /\ edge(c, b)

      @goal
      rel goal() = path(0, 1000)
    "#,
    9,
    provenance::min_max_prob::MinMaxProbProvenance::new(),
    RuntimeEnvironmentOptions::default()
      .with_stop_when_goal_non_empty(true)
      .with_default_scheduler(Some(Scheduler::AStar)),
  );
}
