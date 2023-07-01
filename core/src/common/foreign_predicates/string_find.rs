use super::*;

/// String find foreign predicate
///
/// ``` scl
/// extern pred string_find(s: String, pattern: String, begin: usize, end: usize)[bbff]
/// ```
#[derive(Clone)]
pub struct StringFindBBFF;

impl Default for StringFindBBFF {
  fn default() -> Self {
    Self
  }
}

impl StringFindBBFF {
  pub fn new() -> Self {
    Self
  }
}

impl ForeignPredicate for StringFindBBFF {
  fn name(&self) -> String {
    "string_find".to_string()
  }

  fn arity(&self) -> usize {
    4
  }

  fn argument_type(&self, i: usize) -> ValueType {
    match i {
      0 | 1 => ValueType::String,
      2 | 3 => ValueType::USize,
      _ => panic!("Invalid argument ID `{}`", i),
    }
  }

  fn num_bounded(&self) -> usize {
    2
  }

  fn evaluate(&self, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    assert_eq!(bounded.len(), 2);
    match (&bounded[0], &bounded[1]) {
      (Value::String(s), Value::String(pattern)) => {
        let len: usize = pattern.chars().count(); // .len() doesn't work for non-ASCII str
        s.match_indices(pattern.as_str())
          .map(|(i, _)| (DynamicInputTag::None, vec![Value::from(i), Value::from(i + len)]))
          .collect()
      }
      _ => panic!("Bounded arguments are not strings"),
    }
  }
}
