//! The floating point equality predicate

use super::*;

#[derive(Clone, Debug)]
pub struct NewTagVariable;

impl NewTagVariable {
  pub fn new() -> Self {
    Self
  }
}

impl ForeignPredicate for NewTagVariable {
  fn name(&self) -> String {
    "new_tag_variable".to_string()
  }

  fn generic_type_parameters(&self) -> Vec<ValueType> {
    vec![]
  }

  fn arity(&self) -> usize {
    0
  }

  #[allow(unused)]
  fn argument_type(&self, i: usize) -> ValueType {
    unreachable!("Shouldn't be called as there is no argument")
  }

  fn num_bounded(&self) -> usize {
    0
  }

  #[allow(unused)]
  fn evaluate(&self, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    vec![
      (DynamicInputTag::NewVariable, vec![]),
    ]
  }
}
