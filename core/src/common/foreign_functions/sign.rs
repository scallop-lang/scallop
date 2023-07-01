use super::*;

/// Sign foreign function
///
/// ``` scl
/// extern fn $sign<T: Number>(x: T) -> T
/// ```
#[derive(Clone)]
pub struct Sign;

impl ForeignFunction for Sign {
  fn name(&self) -> String {
    "sign".to_string()
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
      Value::I8(f) => Some(Value::I8(f.signum())),
      Value::I16(f) => Some(Value::I16(f.signum())),
      Value::I32(f) => Some(Value::I32(f.signum())),
      Value::I64(f) => Some(Value::I64(f.signum())),
      Value::I128(f) => Some(Value::I128(f.signum())),
      Value::ISize(f) => Some(Value::ISize(f.signum())),

      Value::U8(_) => Some(Value::U8(1)),
      Value::U16(_) => Some(Value::U16(1)),
      Value::U32(_) => Some(Value::U32(1)),
      Value::U64(_) => Some(Value::U64(1)),
      Value::U128(_) => Some(Value::U128(1)),
      Value::USize(_) => Some(Value::USize(1)),

      Value::F32(f) => {
        if f < 0.0 {
          Some(Value::I32(-1))
        } else if f > 0.0 {
          Some(Value::I32(1))
        } else {
          Some(Value::I32(0))
        }
      }
      Value::F64(f) => {
        if f < 0.0 {
          Some(Value::I32(-1))
        } else if f > 0.0 {
          Some(Value::I32(1))
        } else {
          Some(Value::I32(0))
        }
      }

      _ => panic!("should not happen; input variable to sign should be a number"),
    }
  }
}
