use crate::common::input_tag::*;

use super::*;

impl StaticInputTag for bool {
  fn from_dynamic_input_tag(t: &DynamicInputTag) -> Option<Self> {
    match t {
      DynamicInputTag::Bool(b) => Some(b.clone()),
      _ => None,
    }
  }
}

impl ConvertFromInputTag<()> for bool {
  fn from_input_tag(_: ()) -> Option<Self> { None }
}

impl ConvertFromInputTag<bool> for bool {
  fn from_input_tag(t: bool) -> Option<Self> { Some(t) }
}

impl ConvertFromInputTag<usize> for bool {
  fn from_input_tag(t: usize) -> Option<Self> { Some(t > 0) }
}

impl ConvertFromInputTag<f32> for bool {
  fn from_input_tag(t: f32) -> Option<Self> { Some(t > 0.0) }
}

impl ConvertFromInputTag<f64> for bool {
  fn from_input_tag(t: f64) -> Option<Self> { Some(t > 0.0) }
}

impl<T: Clone + 'static> ConvertFromInputTag<InputDiffProb<T>> for bool {
  fn from_input_tag(t: InputDiffProb<T>) -> Option<Self> { Some(t.0 > 0.0) }
}
