use chrono::Datelike;

use super::*;

/// Get the year (signed integer) in the calendar date
///
/// ``` scl
/// extern fn $datetime_year(d: DateTime) -> i32
/// ```
#[derive(Clone)]
pub struct DateTimeYear;

impl ForeignFunction for DateTimeYear {
  fn name(&self) -> String {
    "datetime_year".to_string()
  }

  fn num_static_arguments(&self) -> usize {
    1
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    assert_eq!(i, 0);
    ForeignFunctionParameterType::BaseType(ValueType::DateTime)
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::BaseType(ValueType::I32)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    match &args[0] {
      Value::DateTime(d) => Some(Value::I32(d.year())),
      _ => None,
    }
  }
}
