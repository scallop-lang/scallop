use super::*;

/// String length
///
/// ``` scl
/// extern fn $string_length(s: String) -> usize
/// ```
#[derive(Clone)]
pub struct StringLength;

impl ForeignFunction for StringLength {
  fn name(&self) -> String {
    "string_length".to_string()
  }

  fn num_static_arguments(&self) -> usize {
    1
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    assert_eq!(i, 0);
    ForeignFunctionParameterType::BaseType(ValueType::String)
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::BaseType(ValueType::USize)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    match &args[0] {
      Value::String(s) => Some(Value::USize(s.len())),
      Value::Str(s) => Some(Value::USize(s.len())),
      _ => None,
    }
  }
}
