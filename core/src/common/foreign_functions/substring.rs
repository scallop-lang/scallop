use super::*;

/// Substring
///
/// ``` scl
/// extern fn $substring(s: String, begin: usize, end: usize?) -> String
/// ```
#[derive(Clone)]
pub struct Substring;

impl ForeignFunction for Substring {
  fn name(&self) -> String {
    "substring".to_string()
  }

  fn num_static_arguments(&self) -> usize {
    2
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    match i {
      0 => ForeignFunctionParameterType::BaseType(ValueType::String),
      1 => ForeignFunctionParameterType::BaseType(ValueType::USize),
      _ => panic!("No argument {}", i),
    }
  }

  fn num_optional_arguments(&self) -> usize {
    1
  }

  fn optional_argument_type(&self, _: usize) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::BaseType(ValueType::USize)
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::BaseType(ValueType::String)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    if args.len() == 2 {
      match (&args[0], &args[1]) {
        (Value::String(s), Value::USize(i)) => Some(Value::String(s[*i..].to_string())),
        _ => panic!("Invalid arguments"),
      }
    } else {
      match (&args[0], &args[1], &args[2]) {
        (Value::String(s), Value::USize(i), Value::USize(j)) => Some(Value::String(s[*i..*j].to_string())),
        _ => panic!("Invalid arguments"),
      }
    }
  }
}
