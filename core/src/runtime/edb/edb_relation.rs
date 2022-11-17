use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct EDBRelation<Prov: Provenance> {
  pub facts: Vec<EDBFact<Prov>>,
  pub disjunctions: Vec<Vec<usize>>,
}

impl<Prov: Provenance> Default for EDBRelation<Prov> {
  fn default() -> Self {
    Self::new()
  }
}

impl<Prov: Provenance> EDBRelation<Prov> {
  pub fn new() -> Self {
    Self {
      facts: vec![],
      disjunctions: vec![],
    }
  }

  pub fn add_disjunction(&mut self, disjunction: Vec<usize>) {
    self.disjunctions.push(disjunction)
  }

  pub fn extend_disjunctions<I>(&mut self, disjunctions: I)
  where
    I: Iterator<Item = Vec<usize>>,
  {
    self.disjunctions.extend(disjunctions)
  }

  pub fn extend_facts(&mut self, new_facts: Vec<EDBFact<Prov>>) {
    self.facts.extend(new_facts)
  }
}
