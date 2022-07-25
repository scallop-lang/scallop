use std::collections::*;

use super::*;

#[derive(Clone, Debug, Default)]
pub struct Disjunctions {
  disjunctions: Vec<Disjunction>,
}

impl Disjunctions {
  pub fn new() -> Self {
    Self {
      disjunctions: Vec::new(),
    }
  }

  pub fn is_empty(&self) -> bool {
    self.disjunctions.is_empty()
  }

  pub fn has_conflict(&self, facts: &BTreeSet<usize>) -> bool {
    // Short hand
    if facts.len() < 2 {
      return false;
    }

    // Check conflict for each disjunction
    for disj in &self.disjunctions {
      if disj.has_conflict(facts) {
        return true;
      }
    }
    false
  }

  pub fn add_disjunction<I>(&mut self, i: I)
  where
    I: Iterator<Item = usize>,
  {
    self.disjunctions.push(i.collect())
  }
}
