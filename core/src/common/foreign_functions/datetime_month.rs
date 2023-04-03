use chrono::Datelike;

use super::*;

/// Get the month of the year starting from 1
///
/// ``` scl
/// extern fn $datetime_month(d: DateTime) -> u32
/// ```
#[derive(Clone)]
pub struct DateTimeMonth;

impl ForeignFunction for DateTimeMonth {
  fn name(&self) -> String {
    "datetime_month".to_string()
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
      Value::DateTime(d) => Some(Value::U32(d.month())),
      _ => None,
    }
  }
}
