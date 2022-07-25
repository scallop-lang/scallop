use crate::runtime::provenance::*;

use super::*;

pub struct EDBRelation<C: ProvenanceContext> {
  pub facts: Vec<EDBFact<C>>,
  pub disjunctions: Vec<Vec<usize>>,
}

impl<C: ProvenanceContext> Default for EDBRelation<C> {
  fn default() -> Self {
    Self::new()
  }
}

impl<C: ProvenanceContext> EDBRelation<C> {
  pub fn new() -> Self {
    Self {
      facts: vec![],
      disjunctions: vec![],
    }
  }

  pub fn add_disjunction(&mut self, disjunction: Vec<usize>) {
    self.disjunctions.push(disjunction)
  }

  pub fn extend_facts(&mut self, new_facts: Vec<EDBFact<C>>) {
    self.facts.extend(new_facts)
  }
}
