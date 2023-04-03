use super::*;

/// String char at
///
/// ``` scl
/// extern fn $string_chat_at(s: String, i: usize) -> char
/// ```
#[derive(Clone)]
pub struct StringCharAt;

impl ForeignFunction for StringCharAt {
  fn name(&self) -> String {
    "string_char_at".to_string()
  }

  fn num_static_arguments(&self) -> usize {
    2
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    match i {
      0 => ForeignFunctionParameterType::BaseType(ValueType::String),
      1 => ForeignFunctionParameterType::BaseType(ValueType::USize),
      _ => panic!("Invalid {}-th argument", i),
    }
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::BaseType(ValueType::Char)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    match (&args[0], &args[1]) {
      (Value::String(s), Value::USize(i)) => s.chars().skip(*i).next().map(Value::Char),
      (Value::Str(s), Value::USize(i)) => s.chars().skip(*i).next().map(Value::Char),
      _ => None,
    }
  }
}
