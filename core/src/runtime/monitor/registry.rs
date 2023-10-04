use std::collections::*;

use crate::runtime::provenance::*;

use super::*;

pub struct MonitorRegistry<Prov: Provenance> {
  monitors: HashMap<String, Box<dyn Monitor<Prov>>>,
}

impl<Prov: Provenance> MonitorRegistry<Prov> {
  pub fn new() -> Self {
    Self {
      monitors: HashMap::new(),
    }
  }

  pub fn std() -> Self {
    let mut m = Self::new();
    m.register(DebugRuntimeMonitor);
    m.register(DebugTagsMonitor);
    m.register(LoggingMonitor);
    m.register(DumpProofsMonitor::new());
    m
  }

  pub fn register<M: Monitor<Prov>>(&mut self, m: M) {
    self.monitors.entry(m.name().to_string()).or_insert(Box::new(m));
  }

  pub fn get(&self, name: &str) -> Option<&Box<dyn Monitor<Prov>>> {
    self.monitors.get(name)
  }

  pub fn load_monitors(&self, names: &[&str]) -> DynamicMonitors<Prov> {
    DynamicMonitors {
      monitors: names
        .iter()
        .filter_map(|name| self.get(name).map(|m| dyn_clone::clone_box(&**m)))
        .collect(),
    }
  }
}
