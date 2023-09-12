use super::*;

/// Get the shape of the tensor
///
/// ``` scl
/// extern pred tensor_shape(bound x: Tensor, dim: usize, size: i64)
/// ```
#[derive(Clone)]
pub struct TensorShape;

impl Default for TensorShape {
  fn default() -> Self {
    Self
  }
}

impl TensorShape {
  pub fn new() -> Self {
    Self
  }
}

impl ForeignPredicate for TensorShape {
  fn name(&self) -> String {
    "tensor_shape".to_string()
  }

  fn arity(&self) -> usize {
    3
  }

  fn argument_type(&self, i: usize) -> ValueType {
    match i {
      0 => ValueType::Tensor,
      1 => ValueType::USize,
      2 => ValueType::I64,
      _ => panic!("Invalid argument ID `{}`", i),
    }
  }

  fn num_bounded(&self) -> usize {
    1
  }

  fn evaluate(&self, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    assert_eq!(bounded.len(), 1);
    match &bounded[0] {
      Value::TensorValue(tensor_value) => tensor_value
        .shape
        .shape()
        .enumerate()
        .map(|(i, size)| {
          let tag = DynamicInputTag::None;
          let tup = vec![Value::from(i), Value::from(size)];
          (tag, tup)
        })
        .collect(),
      _ => panic!("Bounded argument is not a DateTime instance"),
    }
  }
}
