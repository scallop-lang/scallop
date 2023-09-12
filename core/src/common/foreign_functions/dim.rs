use super::*;

/// Dim foreign function, getting dimension of a tensor
///
/// ``` scl
/// extern fn $dim(x: Tensor) -> usize
/// ```
#[derive(Clone)]
pub struct Dim;

impl ForeignFunction for Dim {
  fn name(&self) -> String {
    "dim".to_string()
  }

  fn num_static_arguments(&self) -> usize {
    1
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    assert_eq!(i, 0);
    ForeignFunctionParameterType::BaseType(ValueType::Tensor)
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::BaseType(ValueType::USize)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    match &args[0] {
      Value::TensorValue(t) => Some(Value::USize(t.shape.dim())),
      _ => None,
    }
  }
}
