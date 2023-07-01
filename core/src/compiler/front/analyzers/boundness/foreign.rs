use std::collections::*;

use crate::common::foreign_predicate::*;

#[derive(Clone, Debug)]
pub struct ForeignPredicateBindings {
  bindings: HashMap<String, BindingPattern>,
}

impl ForeignPredicateBindings {
  pub fn contains(&self, name: &str) -> bool {
    self.bindings.contains_key(name)
  }

  pub fn add<F: ForeignPredicate>(&mut self, fp: &F) {
    self.bindings.insert(fp.internal_name(), fp.binding_pattern());
  }

  pub fn get(&self, name: &str) -> Option<&BindingPattern> {
    self.bindings.get(name)
  }
}

impl From<&ForeignPredicateRegistry> for ForeignPredicateBindings {
  fn from(registry: &ForeignPredicateRegistry) -> Self {
    let bindings = registry
      .iter()
      .map(|(name, pred)| (name.clone(), pred.binding_pattern()))
      .collect();
    Self { bindings }
  }
}
