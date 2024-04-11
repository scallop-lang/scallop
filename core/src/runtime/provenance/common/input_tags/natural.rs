use crate::common::foreign_tensor::*;
use crate::common::input_tag::*;

use super::*;

impl StaticInputTag for usize {
  fn from_dynamic_input_tag(t: &DynamicInputTag) -> Option<Self> {
    match t {
      DynamicInputTag::None => Some(1),
      DynamicInputTag::Exclusive(_) => Some(1),
      DynamicInputTag::Natural(n) => Some(*n),
      DynamicInputTag::Bool(b) => Some(if *b { 1 } else { 0 }),
      DynamicInputTag::Float(f) => Some(if *f > 0.0 { 1 } else { 0 }),
      DynamicInputTag::ExclusiveFloat(f, _) => Some(if *f > 0.0 { 1 } else { 0 }),
      DynamicInputTag::FloatWithID(_, f) => Some(if *f > 0.0 { 1 } else { 0 }),
      DynamicInputTag::ExclusiveFloatWithID(_, f, _) => Some(if *f > 0.0 { 1 } else { 0 }),
      DynamicInputTag::Tensor(_) => Some(1),
    }
  }
}

impl ConvertFromInputTag<()> for usize {
  fn from_input_tag(_: ()) -> Option<Self> {
    None
  }
}

impl ConvertFromInputTag<bool> for usize {
  fn from_input_tag(t: bool) -> Option<Self> {
    Some(if t { 1 } else { 0 })
  }
}

impl ConvertFromInputTag<usize> for usize {
  fn from_input_tag(t: usize) -> Option<Self> {
    Some(t)
  }
}

impl ConvertFromInputTag<f32> for usize {
  fn from_input_tag(t: f32) -> Option<Self> {
    Some(if t > 0.0 { 1 } else { 0 })
  }
}

impl ConvertFromInputTag<f64> for usize {
  fn from_input_tag(t: f64) -> Option<Self> {
    Some(if t > 0.0 { 1 } else { 0 })
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputDiffProb<T>> for usize {
  fn from_input_tag(t: InputDiffProb<T>) -> Option<Self> {
    Some(if t.0 > 0.0 { 1 } else { 0 })
  }
}
