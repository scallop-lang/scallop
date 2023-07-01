use super::*;

/// String index_of foreign function
///
/// ``` scl
/// extern fn $string_index_of(s: String, sub: String) -> usize
/// ```
#[derive(Clone)]
pub struct StringIndexOf;

impl ForeignFunction for StringIndexOf {
  fn name(&self) -> String {
    "string_index_of".to_string()
  }

  fn num_static_arguments(&self) -> usize {
    2
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    match i {
      0 | 1 => ForeignFunctionParameterType::BaseType(ValueType::String),
      _ => panic!("Invalid {}-th argument", i),
    }
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::BaseType(ValueType::USize)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    assert_eq!(args.len(), 2);
    match (&args[0], &args[1]) {
      (Value::String(s), Value::String(sub)) => {
        match s.find(sub) {
          Some(index) => Some(Value::USize(index)),
          None => None, // return None if no match found
        }
      }
      _ => panic!("Invalid arguments, expected strings"),
    }
  }
}
