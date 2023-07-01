use super::*;

/// String upper foreign function
///
/// ``` scl
/// extern fn $string_upper(s: String) -> String
/// ```
#[derive(Clone)]
pub struct StringUpper;

impl ForeignFunction for StringUpper {
  fn name(&self) -> String {
    "string_upper".to_string()
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
      Value::String(s) => Some(Value::from(s.to_ascii_uppercase())),
      _ => panic!("Invalid argument, expected string"),
    }
  }
}
