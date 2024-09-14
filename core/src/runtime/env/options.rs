use crate::common::constants::*;

use super::*;

/// The options to create a runtime environment
#[derive(Clone, Debug)]
pub struct RuntimeEnvironmentOptions {
  pub random_seed: u64,
  pub early_discard: bool,
  pub iter_limit: Option<usize>,
  pub stop_when_goal_non_empty: bool,
  pub default_scheduler: Option<Scheduler>,
}

impl Default for RuntimeEnvironmentOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl RuntimeEnvironmentOptions {
  pub fn new() -> Self {
    Self {
      random_seed: DEFAULT_RANDOM_SEED,
      early_discard: true,
      iter_limit: None,
      stop_when_goal_non_empty: false,
      default_scheduler: None,
    }
  }

  pub fn with_stop_when_goal_non_empty(mut self, stop: bool) -> Self {
    self.stop_when_goal_non_empty = stop;
    self
  }

  pub fn with_default_scheduler(mut self, scheduler: Option<Scheduler>) -> Self {
    self.default_scheduler = scheduler;
    self
  }

  pub fn build(self) -> RuntimeEnvironment {
    RuntimeEnvironment::new_from_options(self)
  }
}
