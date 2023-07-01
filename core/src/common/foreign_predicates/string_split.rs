use super::*;

/// String split foreign predicate
///
/// ``` scl
/// extern pred string_split(s: String, pattern: String, output: String)[bbf]
/// ```
#[derive(Clone)]
pub struct StringSplitBBF;

impl Default for StringSplitBBF {
  fn default() -> Self {
    Self
  }
}

impl StringSplitBBF {
  pub fn new() -> Self {
    Self
  }
}

impl ForeignPredicate for StringSplitBBF {
  fn name(&self) -> String {
    "string_split".to_string()
  }

  fn arity(&self) -> usize {
    3
  }

  fn argument_type(&self, i: usize) -> ValueType {
    match i {
      0 | 1 | 2 => ValueType::String,
      _ => panic!("Invalid argument ID `{}`", i),
    }
  }

  fn num_bounded(&self) -> usize {
    2
  }

  fn evaluate(&self, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    assert_eq!(bounded.len(), 2);
    match (&bounded[0], &bounded[1]) {
      (Value::String(s), Value::String(pattern)) => s
        .split(pattern.as_str())
        .map(|part| (DynamicInputTag::None, vec![Value::from(part.to_string())]))
        .collect(),
      _ => panic!("Bounded arguments are not strings"),
    }
  }
}
