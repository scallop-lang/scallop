use chrono::Datelike;

use super::*;

/// Get the month of the year starting from 0
///
/// ``` scl
/// extern fn $datetime_month0(d: DateTime) -> u32
/// ```
#[derive(Clone)]
pub struct DateTimeMonth0;

impl ForeignFunction for DateTimeMonth0 {
  fn name(&self) -> String {
    "datetime_month0".to_string()
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
      Value::DateTime(d) => Some(Value::U32(d.month0())),
      _ => None,
    }
  }
}
