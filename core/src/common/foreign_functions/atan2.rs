use super::*;

/// Arctangent2 foreign function
///
/// ``` scl
/// extern fn $atan2<T: Float>(y: T, x: T) -> T
/// ```
#[derive(Clone)]
pub struct Atan2;

impl ForeignFunction for Atan2 {
  fn name(&self) -> String {
    "atan2".to_string()
  }

  fn num_generic_types(&self) -> usize {
    1
  }

  fn generic_type_family(&self, i: usize) -> TypeFamily {
    match i {
      0 => TypeFamily::Float,
      _ => panic!("No argument {}", i),
    }
  }

  fn num_static_arguments(&self) -> usize {
    2
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    match i {
      0 | 1 => ForeignFunctionParameterType::Generic(0),
      _ => panic!("No argument {}", i),
    }
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::Generic(0)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    match args[0] {
      Value::F32(y) => match args[1] {
        Value::F32(x) => Some(Value::F32(y.atan2(x))),
        _ => panic!("Invalid arguments, should be floats of same bitsize"),
      },
      Value::F64(y) => match args[1] {
        Value::F64(x) => Some(Value::F64(y.atan2(x))),
        _ => panic!("Invalid arguments, should be floats of same bitsize"),
      },
      _ => panic!("Invalid arguments, should be floats of same bitsize"),
    }
  }
}
