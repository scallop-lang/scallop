use crate::common::foreign_tensor::*;
use crate::common::input_tag::*;

use super::*;

#[derive(Clone, Debug)]
pub enum Exclusion {
  Independent,
  Exclusive(usize),
}

impl StaticInputTag for Exclusion {
  fn from_dynamic_input_tag(t: &DynamicInputTag) -> Option<Self> {
    match t {
      DynamicInputTag::Exclusive(e) => Some(Self::Exclusive(e.clone())),
      DynamicInputTag::ExclusiveFloat(_, e) => Some(Self::Exclusive(e.clone())),
      _ => Some(Self::Independent),
    }
  }
}

impl ConvertFromInputTag<()> for Exclusion {
  fn from_input_tag(_: ()) -> Option<Self> {
    None
  }
}

impl ConvertFromInputTag<bool> for Exclusion {
  fn from_input_tag(_: bool) -> Option<Self> {
    None
  }
}

impl ConvertFromInputTag<usize> for Exclusion {
  fn from_input_tag(_: usize) -> Option<Self> {
    None
  }
}

impl ConvertFromInputTag<f64> for Exclusion {
  fn from_input_tag(_: f64) -> Option<Self> {
    None
  }
}

impl ConvertFromInputTag<Exclusion> for Exclusion {
  fn from_input_tag(e: Exclusion) -> Option<Self> {
    Some(e.clone())
  }
}

impl ConvertFromInputTag<InputExclusiveProb> for Exclusion {
  fn from_input_tag(t: InputExclusiveProb) -> Option<Self> {
    match &t.exclusion {
      Some(e) => Some(Self::Exclusive(e.clone())),
      None => Some(Self::Independent),
    }
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputDiffProb<T>> for Exclusion {
  fn from_input_tag(_: InputDiffProb<T>) -> Option<Self> {
    None
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputExclusiveDiffProb<T>> for Exclusion {
  fn from_input_tag(t: InputExclusiveDiffProb<T>) -> Option<Self> {
    match &t.exclusion {
      Some(e) => Some(Self::Exclusive(e.clone())),
      None => Some(Self::Independent),
    }
  }
}
