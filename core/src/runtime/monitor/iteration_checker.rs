use crate::runtime::provenance::Provenance;

use super::*;

/// Iteration Checker
///
/// A monitor with iteration limit; will panic if the execution uses an iteration
/// over the limit
pub struct IterationCheckingMonitor {
  iter_limit: usize,
}

impl IterationCheckingMonitor {
  pub fn new(iter_limit: usize) -> Self {
    Self { iter_limit }
  }
}

impl<Prov: Provenance> Monitor<Prov> for IterationCheckingMonitor {
  fn observe_stratum_iteration(&self, iteration_count: usize) {
    if iteration_count > self.iter_limit {
      panic!(
        "Iteration number ({}) passed over expected limit ({})",
        iteration_count, self.iter_limit
      );
    }
  }
}
