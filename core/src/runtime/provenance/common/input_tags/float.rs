use crate::common::input_tag::*;
use crate::common::tensors::*;

use super::*;

impl StaticInputTag for f64 {
  fn from_dynamic_input_tag(t: &DynamicInputTag) -> Option<Self> {
    match t {
      DynamicInputTag::Float(f) => Some(f.clone()),
      DynamicInputTag::ExclusiveFloat(f, _) => Some(f.clone()),
      _ => None,
    }
  }
}

impl ConvertFromInputTag<()> for f64 {
  fn from_input_tag(_: ()) -> Option<Self> {
    None
  }
}

impl ConvertFromInputTag<bool> for f64 {
  fn from_input_tag(t: bool) -> Option<Self> {
    Some(if t { 1.0 } else { 0.0 })
  }
}

impl ConvertFromInputTag<usize> for f64 {
  fn from_input_tag(t: usize) -> Option<Self> {
    Some(if t > 0 { 1.0 } else { 0.0 })
  }
}

impl ConvertFromInputTag<Exclusion> for f64 {
  fn from_input_tag(_: Exclusion) -> Option<Self> {
    None
  }
}

impl ConvertFromInputTag<f64> for f64 {
  fn from_input_tag(t: f64) -> Option<Self> {
    Some(t)
  }
}

impl ConvertFromInputTag<InputExclusiveProb> for f64 {
  fn from_input_tag(t: InputExclusiveProb) -> Option<Self> {
    Some(t.prob)
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputDiffProb<T>> for f64 {
  fn from_input_tag(t: InputDiffProb<T>) -> Option<Self> {
    Some(t.0)
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputExclusiveDiffProb<T>> for f64 {
  fn from_input_tag(t: InputExclusiveDiffProb<T>) -> Option<Self> {
    Some(t.prob)
  }
}
