use crate::common::foreign_tensor::*;
use crate::common::input_tag::*;

use super::*;

#[derive(Clone)]
pub struct InputExclusiveDiffProbWithID<T: FromTensor> {
  /// The probability of the tag
  pub prob: f64,

  /// The ID of the tag
  pub id: usize,

  /// The external tag for differentiability
  pub external_tag: Option<T>,

  /// An optional identifier of the mutual exclusion
  pub exclusion: Option<usize>,
}

impl<T: FromTensor> InputExclusiveDiffProbWithID<T> {
  pub fn new(id: usize, prob: f64, tag: T, exclusion: Option<usize>) -> Self {
    Self {
      id,
      prob,
      external_tag: Some(tag),
      exclusion,
    }
  }

  pub fn new_without_gradient(id: usize, prob: f64, exclusion: Option<usize>) -> Self {
    Self {
      id,
      prob,
      external_tag: None,
      exclusion,
    }
  }
}

impl<T: FromTensor> std::fmt::Debug for InputExclusiveDiffProbWithID<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.prob.fmt(f)
  }
}

impl<T: FromTensor> From<(usize, f64, T, Option<usize>)> for InputExclusiveDiffProbWithID<T> {
  fn from((id, prob, tag, exclusion): (usize, f64, T, Option<usize>)) -> Self {
    Self {
      id,
      prob,
      external_tag: Some(tag),
      exclusion,
    }
  }
}

impl<T: FromTensor> StaticInputTag for InputExclusiveDiffProbWithID<T> {
  fn from_dynamic_input_tag(t: &DynamicInputTag) -> Option<Self> {
    match t {
      DynamicInputTag::None => None,
      DynamicInputTag::NewVariable => None,
      DynamicInputTag::Exclusive(_) => None,
      DynamicInputTag::Bool(_) => None,
      DynamicInputTag::Natural(_) => None,
      DynamicInputTag::Float(_) => None,
      DynamicInputTag::ExclusiveFloat(_, _) => None,
      DynamicInputTag::FloatWithID(id, prob) => Some(Self {
        id: id.clone(),
        prob: prob.clone(),
        external_tag: None,
        exclusion: None,
      }),
      DynamicInputTag::ExclusiveFloatWithID(id, prob, i) => Some(Self {
        id: id.clone(),
        prob: prob.clone(),
        external_tag: None,
        exclusion: Some(i.clone()),
      }),
      DynamicInputTag::Tensor(_) => None,
    }
  }
}

impl<T: FromTensor> ConvertFromInputTag<()> for InputExclusiveDiffProbWithID<T> {
  fn from_input_tag(_: ()) -> Option<Self> {
    None
  }
}

impl<T: FromTensor> ConvertFromInputTag<bool> for InputExclusiveDiffProbWithID<T> {
  fn from_input_tag(_: bool) -> Option<Self> {
    None
  }
}

impl<T: FromTensor> ConvertFromInputTag<usize> for InputExclusiveDiffProbWithID<T> {
  fn from_input_tag(_: usize) -> Option<Self> {
    None
  }
}

impl<T: FromTensor> ConvertFromInputTag<Exclusion> for InputExclusiveDiffProbWithID<T> {
  fn from_input_tag(_: Exclusion) -> Option<Self> {
    None
  }
}

impl<T: FromTensor> ConvertFromInputTag<f64> for InputExclusiveDiffProbWithID<T> {
  fn from_input_tag(_: f64) -> Option<Self> {
    None
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputExclusiveProb> for InputExclusiveDiffProbWithID<T> {
  fn from_input_tag(_: InputExclusiveProb) -> Option<Self> {
    None
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputDiffProb<T>> for InputExclusiveDiffProbWithID<T> {
  fn from_input_tag(_: InputDiffProb<T>) -> Option<Self> {
    None
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputExclusiveDiffProbWithID<T>> for InputExclusiveDiffProbWithID<T> {
  fn from_input_tag(_: InputExclusiveDiffProbWithID<T>) -> Option<Self> {
    None
  }
}
