use crate::common::input_tag::*;
use crate::common::tensors::*;

use super::*;

impl StaticInputTag for () {
  fn from_dynamic_input_tag(_: &DynamicInputTag) -> Option<Self> {
    Some(())
  }
}

impl ConvertFromInputTag<()> for () {
  fn from_input_tag(_: ()) -> Option<Self> {
    Some(())
  }
}

impl ConvertFromInputTag<bool> for () {
  fn from_input_tag(_: bool) -> Option<Self> {
    Some(())
  }
}

impl ConvertFromInputTag<usize> for () {
  fn from_input_tag(_: usize) -> Option<Self> {
    Some(())
  }
}

impl ConvertFromInputTag<f64> for () {
  fn from_input_tag(_: f64) -> Option<Self> {
    Some(())
  }
}

impl ConvertFromInputTag<Exclusion> for () {
  fn from_input_tag(_: Exclusion) -> Option<Self> {
    Some(())
  }
}

impl ConvertFromInputTag<InputExclusiveProb> for () {
  fn from_input_tag(_: InputExclusiveProb) -> Option<Self> {
    Some(())
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputDiffProb<T>> for () {
  fn from_input_tag(_: InputDiffProb<T>) -> Option<Self> {
    Some(())
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputExclusiveDiffProb<T>> for () {
  fn from_input_tag(_: InputExclusiveDiffProb<T>) -> Option<Self> {
    Some(())
  }
}
