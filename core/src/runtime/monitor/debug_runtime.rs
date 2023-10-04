use crate::runtime::provenance::Provenance;

use super::*;

#[derive(Clone)]
pub struct DebugRuntimeMonitor;

impl<Prov: Provenance> Monitor<Prov> for DebugRuntimeMonitor {
  fn name(&self) -> &'static str {
    "debug-runtime"
  }

  fn observe_executing_stratum(&self, stratum_id: usize) {
    println!("[Executing Stratum #{}]", stratum_id)
  }

  fn observe_stratum_iteration(&self, iteration_count: usize) {
    println!("[Iteration #{}]", iteration_count)
  }
}
