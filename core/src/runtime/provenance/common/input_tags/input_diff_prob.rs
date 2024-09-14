use crate::common::foreign_tensor::*;
use crate::common::input_tag::*;

use super::*;

/// An input differentiable probability.
///
/// It contains two elements.
/// The first is an `f64` which represents the probability of the tag.
/// The second is an `Option<T>` which is the original differentiable object.
/// Note that if the second element is provided as `None` then it means we
/// do not treat the object as differentiable and thus we do not need to
/// back-propagate gradients into it.
/// In such case the probability is treated as a constant.
#[derive(Clone)]
pub struct InputDiffProb<T: FromTensor>(pub f64, pub Option<T>);

impl<T: FromTensor> std::fmt::Debug for InputDiffProb<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.0.fmt(f)
  }
}

impl<T: FromTensor> From<(f64, Option<T>)> for InputDiffProb<T> {
  fn from((p, t): (f64, Option<T>)) -> Self {
    Self(p, t)
  }
}

impl<T: FromTensor> StaticInputTag for InputDiffProb<T> {
  fn from_dynamic_input_tag(t: &DynamicInputTag) -> Option<Self> {
    match t {
      DynamicInputTag::None => None,
      DynamicInputTag::NewVariable => None,
      DynamicInputTag::Bool(b) => Some(Self(if *b { 1.0 } else { 0.0 }, None)),
      DynamicInputTag::Natural(n) => Some(Self(if *n > 0 { 1.0 } else { 0.0 }, None)),
      DynamicInputTag::Exclusive(_) => None,
      DynamicInputTag::Float(f) => Some(Self(f.clone(), None)),
      DynamicInputTag::ExclusiveFloat(f, _) => Some(Self(f.clone(), None)),
      DynamicInputTag::FloatWithID(_, f) => Some(Self(f.clone(), None)),
      DynamicInputTag::ExclusiveFloatWithID(_, f, _) => Some(Self(f.clone(), None)),
      DynamicInputTag::Tensor(t) => Some(Self(t.get_f64(), T::from_tensor(t.clone()))),
    }
  }
}

impl<T: FromTensor> ConvertFromInputTag<()> for InputDiffProb<T> {
  fn from_input_tag(_: ()) -> Option<Self> {
    None
  }
}

impl<T: FromTensor> ConvertFromInputTag<bool> for InputDiffProb<T> {
  fn from_input_tag(b: bool) -> Option<Self> {
    if b {
      None
    } else {
      Some(Self(0.0, None))
    }
  }
}

impl<T: FromTensor> ConvertFromInputTag<usize> for InputDiffProb<T> {
  fn from_input_tag(u: usize) -> Option<Self> {
    if u > 0 {
      None
    } else {
      Some(Self(0.0, None))
    }
  }
}

impl<T: FromTensor> ConvertFromInputTag<Exclusion> for InputDiffProb<T> {
  fn from_input_tag(_: Exclusion) -> Option<Self> {
    None
  }
}

impl<T: FromTensor> ConvertFromInputTag<f64> for InputDiffProb<T> {
  fn from_input_tag(t: f64) -> Option<Self> {
    Some(Self(t, None))
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputExclusiveProb> for InputDiffProb<T> {
  fn from_input_tag(t: InputExclusiveProb) -> Option<Self> {
    Some(Self(t.prob, None))
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputDiffProb<T>> for InputDiffProb<T> {
  fn from_input_tag(t: InputDiffProb<T>) -> Option<Self> {
    Some(t.clone())
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputExclusiveDiffProb<T>> for InputDiffProb<T> {
  fn from_input_tag(t: InputExclusiveDiffProb<T>) -> Option<Self> {
    Some(Self(t.prob, None))
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputExclusiveDiffProbWithID<T>> for InputDiffProb<T> {
  fn from_input_tag(t: InputExclusiveDiffProbWithID<T>) -> Option<Self> {
    Some(Self(t.prob, t.external_tag))
  }
}
