use std::collections::*;

use crate::common::foreign_predicate::*;
use crate::common::value_type::*;

/// The type of a foreign predicate.
/// Essentially a list of basic types.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PredicateType {
  pub arguments: Vec<ValueType>,
}

impl<P: ForeignPredicate> From<&P> for PredicateType {
  fn from(p: &P) -> Self {
    Self {
      arguments: p.argument_types(),
    }
  }
}

impl std::ops::Index<usize> for PredicateType {
  type Output = ValueType;

  fn index(&self, index: usize) -> &Self::Output {
    &self.arguments[index]
  }
}

impl PredicateType {
  pub fn len(&self) -> usize {
    self.arguments.len()
  }

  pub fn iter<'a>(&'a self) -> std::slice::Iter<'a, ValueType> {
    self.arguments.iter()
  }
}

/// Predicate type registry that stores information about foreign
/// predicates and their types
#[derive(Clone, Debug)]
pub struct PredicateTypeRegistry {
  pub predicate_types: HashMap<String, PredicateType>,
}

impl PredicateTypeRegistry {
  /// Create a new empty predicate type registry
  pub fn empty() -> Self {
    Self {
      predicate_types: HashMap::new(),
    }
  }

  /// Create a new predicate type registry
  pub fn from_foreign_predicate_registry(foreign_predicate_registry: &ForeignPredicateRegistry) -> Self {
    let mut type_registry = Self::empty();
    for (_, fp) in foreign_predicate_registry {
      type_registry.add_foreign_predicate(fp)
    }
    type_registry
  }

  /// Add a new foreign predicate to the predicate type registry
  pub fn add_foreign_predicate<P: ForeignPredicate>(&mut self, p: &P) {
    self.predicate_types.insert(p.internal_name(), PredicateType::from(p));
  }

  /// Check if the registry contains a predicate
  pub fn contains_predicate(&self, p: &str) -> bool {
    self.predicate_types.contains_key(p)
  }

  /// Get a predicate type
  pub fn get(&self, p: &str) -> Option<&PredicateType> {
    self.predicate_types.get(p)
  }
}
