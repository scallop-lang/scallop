use super::*;

/// String concat
///
/// ``` scl
/// extern fn $string_concat(s: String...) -> String
/// ```
#[derive(Clone)]
pub struct StringConcat;

impl ForeignFunction for StringConcat {
  fn name(&self) -> String {
    "string_concat".to_string()
  }

  fn has_variable_arguments(&self) -> bool {
    true
  }

  fn variable_argument_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::BaseType(ValueType::String)
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::BaseType(ValueType::String)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    let mut result = "".to_string();
    for arg in args {
      match arg {
        Value::String(s) => {
          result += &s;
        }
        _ => panic!("Argument is not string"),
      }
    }
    Some(Value::String(result))
  }
}
