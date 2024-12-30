use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct AggregateBodyAttribute {
  pub aggregator: String,
  pub num_group_by_vars: usize,
  pub num_arg_vars: usize,
  pub num_key_vars: usize,
}

impl AggregateBodyAttribute {
  pub fn new(aggregator: String, num_group_by_vars: usize, num_arg_vars: usize, num_key_vars: usize) -> Self {
    Self {
      aggregator,
      num_group_by_vars,
      num_arg_vars,
      num_key_vars,
    }
  }
}

impl AttributeTrait for AggregateBodyAttribute {
  fn name(&self) -> String {
    "aggregate_body".to_string()
  }

  fn args(&self) -> Vec<String> {
    vec![
      format!("num_group_by={}", self.num_group_by_vars),
      format!("num_arg={}", self.num_arg_vars),
      format!("num_key={}", self.num_key_vars),
    ]
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AggregateGroupByAttribute {
  pub num_join_group_by_vars: usize,
  pub num_other_group_by_vars: usize,
}

impl AggregateGroupByAttribute {
  pub fn new(num_join_group_by_vars: usize, num_other_group_by_vars: usize) -> Self {
    Self {
      num_join_group_by_vars,
      num_other_group_by_vars,
    }
  }
}

impl AttributeTrait for AggregateGroupByAttribute {
  fn name(&self) -> String {
    "aggregate_group_by".to_string()
  }

  fn args(&self) -> Vec<String> {
    vec![
      format!("num_joined={}", self.num_join_group_by_vars),
      format!("num_other={}", self.num_other_group_by_vars),
    ]
  }
}
