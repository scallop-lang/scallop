use chrono::Datelike;

use super::*;

/// Get the day of the month starting from 1
///
/// ``` scl
/// extern fn $datetime_day(d: DateTime) -> u32
/// ```
#[derive(Clone)]
pub struct DateTimeDay;

impl ForeignFunction for DateTimeDay {
  fn name(&self) -> String {
    "datetime_day".to_string()
  }

  fn num_static_arguments(&self) -> usize {
    1
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    assert_eq!(i, 0);
    ForeignFunctionParameterType::BaseType(ValueType::DateTime)
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::BaseType(ValueType::U32)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    match &args[0] {
      Value::DateTime(d) => Some(Value::U32(d.day())),
      _ => None,
    }
  }
}
