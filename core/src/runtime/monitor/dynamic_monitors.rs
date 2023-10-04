use crate::common::tuple::*;
use crate::runtime::provenance::*;

use super::*;

pub struct DynamicMonitors<Prov: Provenance> {
  pub monitors: Vec<Box<dyn Monitor<Prov>>>,
}

impl<Prov: Provenance> Clone for DynamicMonitors<Prov> {
  fn clone(&self) -> Self {
    Self {
      monitors: self.monitors.iter().map(|m| dyn_clone::clone_box(&**m)).collect(),
    }
  }
}

impl<Prov: Provenance> DynamicMonitors<Prov> {
  pub fn new() -> Self {
    Self { monitors: vec![] }
  }

  pub fn monitor_names(&self) -> Vec<&str> {
    self.monitors.iter().map(|m| m.name()).collect()
  }

  pub fn is_empty(&self) -> bool {
    self.monitors.is_empty()
  }

  /// Add a new monitor to this list of monitors
  pub fn add<M: Monitor<Prov> + 'static>(&mut self, m: M) {
    self.monitors.push(Box::new(m))
  }

  pub fn with_monitor<M: Monitor<Prov> + 'static>(mut self, m: M) -> Self {
    self.add(m);
    self
  }

  pub fn extend(&mut self, other: Self) {
    for m in other.monitors {
      self.monitors.push(m);
    }
  }
}

macro_rules! dynamic_monitors_observe_event {
  ($func:ident, ($($arg:ident: $ty:ty),*)) => {
    fn $func(&self, $($arg : $ty,)*) {
      for m in &self.monitors {
        m.$func($($arg),*);
      }
    }
  };
}

impl<Prov: Provenance> Monitor<Prov> for DynamicMonitors<Prov> {
  fn name(&self) -> &'static str {
    "multiple"
  }
  dynamic_monitors_observe_event!(observe_executing_stratum, (stratum_id: usize));
  dynamic_monitors_observe_event!(observe_stratum_iteration, (iteration_count: usize));
  dynamic_monitors_observe_event!(observe_hitting_iteration_limit, ());
  dynamic_monitors_observe_event!(observe_converging, ());
  dynamic_monitors_observe_event!(observe_loading_relation, (relation: &str));
  dynamic_monitors_observe_event!(observe_loading_relation_from_edb, (relation: &str));
  dynamic_monitors_observe_event!(observe_loading_relation_from_idb, (relation: &str));
  dynamic_monitors_observe_event!(observe_tagging, (tup: &Tuple, input_tag: &Option<Prov::InputTag>, tag: &Prov::Tag));
  dynamic_monitors_observe_event!(observe_finish_execution, ());
  dynamic_monitors_observe_event!(observe_recovering_relation, (relation: &str));
  dynamic_monitors_observe_event!(observe_recover, (tup: &Tuple, tag: &Prov::Tag, output_tag: &Prov::OutputTag));
  dynamic_monitors_observe_event!(observe_finish_recovering_relation, (relation: &str));
}
