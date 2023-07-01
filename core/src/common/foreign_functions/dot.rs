use super::*;

/// Dot product of two tensors
///
/// ``` scl
/// extern fn $dot(x: Tensor, y: Tensor) -> Tensor
/// ```
#[derive(Clone)]
pub struct Dot;

impl ForeignFunction for Dot {
  fn name(&self) -> String {
    "dot".to_string()
  }

  fn num_static_arguments(&self) -> usize {
    2
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    assert!(i < 2);
    ForeignFunctionParameterType::BaseType(ValueType::Tensor)
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::BaseType(ValueType::Tensor)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    let mut iter = args.into_iter();
    match (iter.next().unwrap(), iter.next().unwrap()) {
      (Value::TensorValue(t1), Value::TensorValue(t2)) => t1.dot(t2).map(Value::TensorValue),
      _ => None,
    }
  }
}
