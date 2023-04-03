use super::*;

/// Absolute value foreign function
///
/// ``` scl
/// extern fn $abs<T: Number>(x: T) -> T
/// ```
#[derive(Clone)]
pub struct Abs;

impl ForeignFunction for Abs {
  fn name(&self) -> String {
    "abs".to_string()
  }

  fn num_generic_types(&self) -> usize {
    1
  }

  fn generic_type_family(&self, i: usize) -> TypeFamily {
    assert_eq!(i, 0);
    TypeFamily::Number
  }

  fn num_static_arguments(&self) -> usize {
    1
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    assert_eq!(i, 0);
    ForeignFunctionParameterType::Generic(0)
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::Generic(0)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    match args[0] {
      // Signed integers, take absolute
      Value::I8(f) => Some(Value::I8(f.abs())),
      Value::I16(f) => Some(Value::I16(f.abs())),
      Value::I32(f) => Some(Value::I32(f.abs())),
      Value::I64(f) => Some(Value::I64(f.abs())),
      Value::I128(f) => Some(Value::I128(f.abs())),
      Value::ISize(f) => Some(Value::ISize(f.abs())),

      // Unsigned integers, directly return
      Value::U8(f) => Some(Value::U8(f)),
      Value::U16(f) => Some(Value::U16(f)),
      Value::U32(f) => Some(Value::U32(f)),
      Value::U64(f) => Some(Value::U64(f)),
      Value::U128(f) => Some(Value::U128(f)),
      Value::USize(f) => Some(Value::USize(f)),

      // Floating points, take absolute
      Value::F32(f) => Some(Value::F32(f.abs())),
      Value::F64(f) => Some(Value::F64(f.abs())),
      _ => panic!("should not happen; input variable to abs should be a number"),
    }
  }
}
