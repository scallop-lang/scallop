use std::sync::*;

use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::common::constants::*;
use crate::common::foreign_function::*;
use crate::common::foreign_predicate::*;
use crate::utils::*;

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

  /// Build a runtime environment from this options
  pub fn build(self) -> RuntimeEnvironment {
    let rng = SmallRng::seed_from_u64(self.random_seed);
    RuntimeEnvironment {
      random_seed: self.random_seed,
      rng: Arc::new(Mutex::new(rng)),
      early_discard: self.early_discard,
      iter_limit: self.iter_limit,
      function_registry: ForeignFunctionRegistry::std(),
      predicate_registry: ForeignPredicateRegistry::std(),
      exclusion_id_allocator: Arc::new(Mutex::new(IdAllocator::new())),
    }
  }
}
