use super::*;

/// String chars foreign predicate
///
/// ``` scl
/// extern pred string_chars(s: String, id: usize, c: char)[bff]
/// ```
#[derive(Clone)]
pub struct StringCharsBFF;

impl Default for StringCharsBFF {
  fn default() -> Self {
    Self
  }
}

impl StringCharsBFF {
  pub fn new() -> Self {
    Self
  }
}

impl ForeignPredicate for StringCharsBFF {
  fn name(&self) -> String {
    "string_chars".to_string()
  }

  fn arity(&self) -> usize {
    3
  }

  fn argument_type(&self, i: usize) -> ValueType {
    match i {
      0 => ValueType::String,
      1 => ValueType::USize,
      2 => ValueType::Char,
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
      Value::String(s) => s
        .chars()
        .enumerate()
        .map(|(i, c)| (DynamicInputTag::None, vec![Value::from(i), Value::from(c)]))
        .collect(),
      _ => panic!("Bounded argument is not string"),
    }
  }
}
