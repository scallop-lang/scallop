use super::*;

/// Power foreign function (x^y)
///
/// ``` scl
/// extern fn $pow<T: Integer>(x: T, y: u32) -> T
/// ```
#[derive(Clone)]
pub struct Pow;

impl ForeignFunction for Pow {
  fn name(&self) -> String {
    "pow".to_string()
  }

  fn num_generic_types(&self) -> usize {
    1
  }

  fn generic_type_family(&self, i: usize) -> TypeFamily {
    assert_eq!(i, 0);
    TypeFamily::Integer
  }

  fn num_static_arguments(&self) -> usize {
    2
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    match i {
      0 => ForeignFunctionParameterType::Generic(0),
      1 => ForeignFunctionParameterType::BaseType(ValueType::U32),
      _ => panic!("No argument {}", i),
    }
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::Generic(0)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    match (&args[0], &args[1]) {
      (Value::I8(x), Value::U32(y)) => Some(Value::I8(x.pow(*y))),
      (Value::I16(x), Value::U32(y)) => Some(Value::I16(x.pow(*y))),
      (Value::I32(x), Value::U32(y)) => Some(Value::I32(x.pow(*y))),
      (Value::I64(x), Value::U32(y)) => Some(Value::I64(x.pow(*y))),
      (Value::I128(x), Value::U32(y)) => Some(Value::I128(x.pow(*y))),
      (Value::ISize(x), Value::U32(y)) => Some(Value::ISize(x.pow(*y))),
      (Value::U8(x), Value::U32(y)) => Some(Value::U8(x.pow(*y))),
      (Value::U16(x), Value::U32(y)) => Some(Value::U16(x.pow(*y))),
      (Value::U32(x), Value::U32(y)) => Some(Value::U32(x.pow(*y))),
      (Value::U64(x), Value::U32(y)) => Some(Value::U64(x.pow(*y))),
      (Value::U128(x), Value::U32(y)) => Some(Value::U128(x.pow(*y))),
      (Value::USize(x), Value::U32(y)) => Some(Value::USize(x.pow(*y))),
      _ => panic!("Invalid arguments"),
    }
  }
}
