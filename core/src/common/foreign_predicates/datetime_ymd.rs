use super::*;
use chrono::Datelike;

/// DateTime YMD extraction foreign predicate
///
/// ``` scl
/// extern pred datetime_ymd(d: DateTime, year: i32, month: u32, date: u32)[bfff]
/// ```
#[derive(Clone)]
pub struct DateTimeYMD;

impl Default for DateTimeYMD {
  fn default() -> Self {
    Self
  }
}

impl DateTimeYMD {
  pub fn new() -> Self {
    Self
  }
}

impl ForeignPredicate for DateTimeYMD {
  fn name(&self) -> String {
    "datetime_ymd".to_string()
  }

  fn arity(&self) -> usize {
    4
  }

  fn argument_type(&self, i: usize) -> ValueType {
    match i {
      0 => ValueType::DateTime,
      1 => ValueType::I32,
      2 | 3 => ValueType::U32,
      _ => panic!("Invalid argument ID `{}`", i),
    }
  }

  fn num_bounded(&self) -> usize {
    1
  }

  fn evaluate(&self, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    assert_eq!(bounded.len(), 1);
    match &bounded[0] {
      Value::DateTime(dt) => {
        vec![(
          DynamicInputTag::None,
          vec![Value::from(dt.year()), Value::from(dt.month()), Value::from(dt.day())],
        )]
      }
      _ => panic!("Bounded argument is not a DateTime instance"),
    }
  }
}
