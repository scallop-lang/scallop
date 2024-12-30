use std::collections::*;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Arc {
  pub left: Vec<usize>,
  pub right: usize,
  pub left_relations: Vec<String>,
  pub bounded_vars: HashSet<Variable>,
  pub is_edb: bool,
}

impl Arc {
  pub fn weight(&self) -> i32 {
    let demand_weight = self.left_relations.iter().filter(|r| r.starts_with("d#")).count() as i32;
    let num_bounded_vars = self.bounded_vars.len() as i32;
    let edb_weight = if self.left.is_empty() && self.is_edb { 1 } else { 0 };
    demand_weight + num_bounded_vars + edb_weight
  }
}
