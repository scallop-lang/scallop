use super::*;

/// String trim foreign function
///
/// ``` scl
/// extern fn $string_trim(s: String) -> String
/// ```
#[derive(Clone)]
pub struct StringTrim;

impl ForeignFunction for StringTrim {
  fn name(&self) -> String {
    "string_trim".to_string()
  }

  fn num_static_arguments(&self) -> usize {
    1
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    assert_eq!(i, 0);
    ForeignFunctionParameterType::BaseType(ValueType::String)
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::BaseType(ValueType::String)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    assert_eq!(args.len(), 1);
    match &args[0] {
      Value::String(s) => Some(Value::from(s.trim().to_string())),
      _ => panic!("Invalid argument, expected string"),
    }
  }
}
