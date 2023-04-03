//! The soft less-than predicate

use super::*;

/// Soft Less Than foreign predicate
///
/// ``` scl
/// extern pred `<`<T: Number>(lhs: T, rhs: T)
/// ```
#[derive(Clone, Debug)]
pub struct SoftNumberLt {
  /// The type of the operands
  pub ty: ValueType,

  /// The function chosen as the sigmoid
  pub sigmoid: SigmoidFunction,
}

impl SoftNumberLt {
  pub fn new(ty: ValueType) -> Self {
    Self {
      ty,
      sigmoid: SigmoidFunction::default()
    }
  }

  pub fn new_with_sigmoid_fn(ty: ValueType, sigmoid: SigmoidFunction) -> Self {
    Self {
      ty,
      sigmoid,
    }
  }

  fn soft_lt<T>(&self, lhs: &Value, rhs: &Value) -> Option<f64>
  where
    T: std::ops::Sub<Output = T> + TryInto<f64>,
    Value: TryInto<T>,
  {
    let lhs: T = lhs.clone().try_into().ok()?;
    let rhs: T = rhs.clone().try_into().ok()?;
    let diff: f64 = (rhs - lhs).try_into().ok()?;
    Some(self.sigmoid.eval(diff))
  }

  fn soft_lt_wrapper<T>(&self, lhs: &Value, rhs: &Value) -> Vec<(DynamicInputTag, Vec<Value>)>
  where
    T: std::ops::Sub<Output = T> + TryInto<f64>,
    Value: TryInto<T>,
  {
    if let Some(prob) = self.soft_lt(lhs, rhs) {
      vec![(DynamicInputTag::Float(prob), vec![])]
    } else {
      vec![]
    }
  }
}

impl ForeignPredicate for SoftNumberLt {
  fn name(&self) -> String {
    format!("soft_lt_{}", self.ty)
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
    match &self.ty {
      ValueType::I8 => self.soft_lt_wrapper::<i8>(lhs, rhs),
      ValueType::I16 => self.soft_lt_wrapper::<i16>(lhs, rhs),
      ValueType::I32 => self.soft_lt_wrapper::<i32>(lhs, rhs),
      // ValueType::I64 => self.soft_lt_wrapper::<i64>(lhs, rhs),
      // ValueType::I128 => self.soft_lt_wrapper::<i128>(lhs, rhs),
      // ValueType::ISize => self.soft_lt_wrapper::<isize>(lhs, rhs),
      ValueType::U8 => self.soft_lt_wrapper::<u8>(lhs, rhs),
      ValueType::U16 => self.soft_lt_wrapper::<u16>(lhs, rhs),
      ValueType::U32 => self.soft_lt_wrapper::<u32>(lhs, rhs),
      // ValueType::U64 => self.soft_lt_wrapper::<u64>(lhs, rhs),
      // ValueType::U128 => self.soft_lt_wrapper::<u128>(lhs, rhs),
      // ValueType::USize => self.soft_lt_wrapper::<usize>(lhs, rhs),
      ValueType::F32 => self.soft_lt_wrapper::<f32>(lhs, rhs),
      ValueType::F64 => self.soft_lt_wrapper::<f64>(lhs, rhs),
      _ => vec![],
    }
  }
}
