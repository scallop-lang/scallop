use super::*;

/// String replace foreign function
///
/// ``` scl
/// extern fn $string_replace(s: String, pat: String, replace: String) -> String
/// ```
#[derive(Clone)]
pub struct StringReplace;

impl ForeignFunction for StringReplace {
  fn name(&self) -> String {
    "string_replace".to_string()
  }

  fn num_static_arguments(&self) -> usize {
    3
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    assert!(i < 3);
    ForeignFunctionParameterType::BaseType(ValueType::String)
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::BaseType(ValueType::String)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    assert_eq!(args.len(), 3);
    match (&args[0], &args[1], &args[2]) {
      (Value::String(s), Value::String(pat), Value::String(replace)) => Some(Value::String(s.replace(pat, replace))),
      _ => panic!("Invalid argument, expected string"),
    }
  }
}
