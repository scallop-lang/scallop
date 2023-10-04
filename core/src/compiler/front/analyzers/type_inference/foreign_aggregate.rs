use std::collections::*;

use crate::common::foreign_aggregate::*;

#[derive(Debug, Clone)]
pub struct AggregateTypeRegistry {
  pub aggregate_types: HashMap<String, AggregateType>,
}

impl AggregateTypeRegistry {
  pub fn empty() -> Self {
    Self {
      aggregate_types: HashMap::new(),
    }
  }

  pub fn from_aggregate_registry(far: &AggregateRegistry) -> Self {
    let mut registry = Self::empty();
    for (_, fa) in far.iter() {
      let name = fa.name();
      let agg_type = fa.aggregate_type();
      registry.aggregate_types.insert(name, agg_type);
    }
    registry
  }

  pub fn get(&self, agg_name: &str) -> Option<&AggregateType> {
    self.aggregate_types.get(agg_name)
  }
}
