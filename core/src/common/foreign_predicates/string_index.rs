use super::*;

/// String index foreign predicate
///
/// ``` scl
/// extern pred string_index(s: String, id: usize)[bf]
/// ```
#[derive(Clone)]
pub struct StringIndexBF;

impl Default for StringIndexBF {
  fn default() -> Self {
    Self
  }
}

impl StringIndexBF {
  pub fn new() -> Self {
    Self
  }
}

impl ForeignPredicate for StringIndexBF {
  fn name(&self) -> String {
    "string_index".to_string()
  }

  fn arity(&self) -> usize {
    2
  }

  fn argument_type(&self, i: usize) -> ValueType {
    match i {
      0 => ValueType::String,
      1 => ValueType::USize,
      _ => panic!("Invalid argument ID `{}`", i),
    }
  }

  fn num_bounded(&self) -> usize {
    1
  }

  fn evaluate(&self, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    assert_eq!(bounded.len(), 1);
    let s = &bounded[0];
    match s {
      Value::String(s) => (0usize..s.len())
        .map(|i| (DynamicInputTag::None, vec![Value::from(i)]))
        .collect(),
      _ => panic!("Bounded argument is not string"),
    }
  }
}
