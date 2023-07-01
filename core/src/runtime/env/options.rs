use crate::common::constants::*;

use super::*;

/// The options to create a runtime environment
#[derive(Clone, Debug)]
pub struct RuntimeEnvironmentOptions {
  pub random_seed: u64,
  pub early_discard: bool,
  pub iter_limit: Option<usize>,
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
    }
  }

  pub fn build(self) -> RuntimeEnvironment {
    RuntimeEnvironment::new_from_options(self)
  }
}
