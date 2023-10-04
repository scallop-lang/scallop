use crate::common::foreign_tensor::*;
use crate::common::input_tag::*;

use super::*;

#[derive(Clone)]
pub struct InputExclusiveProb {
  /// The probability of the tag
  pub prob: f64,

  /// An optional identifier of the mutual exclusion
  pub exclusion: Option<usize>,
}

impl InputExclusiveProb {
  pub fn new(prob: f64, exclusion: Option<usize>) -> Self {
    Self { prob, exclusion }
  }
}

impl std::fmt::Debug for InputExclusiveProb {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.prob.fmt(f)
  }
}

impl From<f64> for InputExclusiveProb {
  fn from(p: f64) -> Self {
    Self {
      prob: p,
      exclusion: None,
    }
  }
}

impl From<(f64, Option<usize>)> for InputExclusiveProb {
  fn from((p, e): (f64, Option<usize>)) -> Self {
    Self { prob: p, exclusion: e }
  }
}

impl From<(f64, usize)> for InputExclusiveProb {
  fn from((p, e): (f64, usize)) -> Self {
    Self {
      prob: p,
      exclusion: Some(e),
    }
  }
}

impl StaticInputTag for InputExclusiveProb {
  fn from_dynamic_input_tag(t: &DynamicInputTag) -> Option<Self> {
    match t {
      DynamicInputTag::Float(f) => Some(Self::new(f.clone(), None)),
      DynamicInputTag::ExclusiveFloat(f, id) => Some(Self::new(f.clone(), Some(id.clone()))),
      DynamicInputTag::Tensor(t) => Some(Self::new(t.get_f64(), None)),
      _ => None,
    }
  }
}

impl ConvertFromInputTag<()> for InputExclusiveProb {
  fn from_input_tag(_: ()) -> Option<Self> {
    None
  }
}

impl ConvertFromInputTag<bool> for InputExclusiveProb {
  fn from_input_tag(t: bool) -> Option<Self> {
    if t {
      None
    } else {
      Some(Self::new(0.0, None))
    }
  }
}

impl ConvertFromInputTag<usize> for InputExclusiveProb {
  fn from_input_tag(t: usize) -> Option<Self> {
    if t > 0 {
      None
    } else {
      Some(Self::new(0.0, None))
    }
  }
}

impl ConvertFromInputTag<f64> for InputExclusiveProb {
  fn from_input_tag(t: f64) -> Option<Self> {
    Some(Self::new(t, None))
  }
}

impl ConvertFromInputTag<Exclusion> for InputExclusiveProb {
  fn from_input_tag(t: Exclusion) -> Option<Self> {
    match t {
      Exclusion::Independent => None,
      Exclusion::Exclusive(id) => Some(Self::new(1.0, Some(id))),
    }
  }
}

impl ConvertFromInputTag<InputExclusiveProb> for InputExclusiveProb {
  fn from_input_tag(t: InputExclusiveProb) -> Option<Self> {
    Some(t.clone())
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputDiffProb<T>> for InputExclusiveProb {
  fn from_input_tag(t: InputDiffProb<T>) -> Option<Self> {
    Some(Self::new(t.0, None))
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputExclusiveDiffProb<T>> for InputExclusiveProb {
  fn from_input_tag(t: InputExclusiveDiffProb<T>) -> Option<Self> {
    Some(Self::new(t.prob.clone(), t.exclusion.clone()))
  }
}
