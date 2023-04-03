//! The floating point equality predicate

use super::*;

#[derive(Clone, Debug)]
pub struct FloatEq {
  /// The floating point type
  pub ty: ValueType,

  /// The type of the operands
  pub threshold: f64,
}

impl FloatEq {
  pub fn new(ty: ValueType) -> Self {
    assert!(ty.is_float());
    Self {
      ty,
      threshold: 0.001,
    }
  }

  pub fn new_with_threshold(ty: ValueType, threshold: f64) -> Self {
    Self {
      ty,
      threshold,
    }
  }
}

impl ForeignPredicate for FloatEq {
  fn name(&self) -> String {
    format!("float_eq_{}", self.ty)
  }

  fn arity(&self) -> usize {
    2
  }

  fn argument_type(&self, i: usize) -> ValueType {
    assert!(i < 2);
    self.ty.clone()
  }

  fn num_bounded(&self) -> usize {
    2
  }

  fn evaluate(&self, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    assert_eq!(bounded.len(), 2);
    let lhs = &bounded[0];
    let rhs = &bounded[1];
    match (&self.ty, lhs, rhs) {
      (ValueType::F32, Value::F32(l), Value::F32(r)) if (l - r).abs() < (self.threshold as f32) => {
        vec![(DynamicInputTag::None, vec![])]
      },
      (ValueType::F64, Value::F64(l), Value::F64(r)) if (l - r).abs() < self.threshold => {
        vec![(DynamicInputTag::None, vec![])]
      },
      _ => vec![],
    }
  }
}
