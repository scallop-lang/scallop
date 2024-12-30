use super::*;

/// Demand attributes to the relations which are on-demand relations
#[derive(Clone, Debug, PartialEq)]
pub struct DemandAttribute {
  pub pattern: String,
}

impl DemandAttribute {
  pub fn new(pattern: String) -> Self {
    Self { pattern }
  }
}

impl AttributeTrait for DemandAttribute {
  fn name(&self) -> String {
    "demand".to_string()
  }

  fn args(&self) -> Vec<String> {
    vec![self.pattern.clone()]
  }
}
