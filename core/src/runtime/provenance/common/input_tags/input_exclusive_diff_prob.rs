use crate::common::input_tag::*;
use crate::common::foreign_tensor::*;

use super::*;

#[derive(Clone)]
pub struct InputExclusiveDiffProb<T: FromTensor> {
  /// The probability of the tag
  pub prob: f64,

  /// The external tag for differentiability
  pub external_tag: Option<T>,

  /// An optional identifier of the mutual exclusion
  pub exclusion: Option<usize>,
}

impl<T: FromTensor> InputExclusiveDiffProb<T> {
  pub fn new(prob: f64, tag: T, exclusion: Option<usize>) -> Self {
    Self {
      prob,
      external_tag: Some(tag),
      exclusion,
    }
  }

  pub fn new_without_gradient(prob: f64, exclusion: Option<usize>) -> Self {
    Self {
      prob,
      external_tag: None,
      exclusion,
    }
  }
}

impl<T: FromTensor> std::fmt::Debug for InputExclusiveDiffProb<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.prob.fmt(f)
  }
}

impl<T: FromTensor> From<(f64, T, Option<usize>)> for InputExclusiveDiffProb<T> {
  fn from((prob, tag, exclusion): (f64, T, Option<usize>)) -> Self {
    Self {
      prob,
      external_tag: Some(tag),
      exclusion,
    }
  }
}

impl<T: FromTensor> StaticInputTag for InputExclusiveDiffProb<T> {
  fn from_dynamic_input_tag(t: &DynamicInputTag) -> Option<Self> {
    match t {
      DynamicInputTag::None => None,
      DynamicInputTag::Bool(b) => Some(Self {
        prob: if *b { 1.0 } else { 0.0 },
        external_tag: None,
        exclusion: None,
      }),
      DynamicInputTag::Natural(n) => Some(Self {
        prob: if *n > 0 { 1.0 } else { 0.0 },
        external_tag: None,
        exclusion: None,
      }),
      DynamicInputTag::Exclusive(i) => Some(Self {
        prob: 1.0,
        external_tag: None,
        exclusion: Some(i.clone()),
      }),
      DynamicInputTag::Float(prob) => Some(Self {
        prob: prob.clone(),
        external_tag: None,
        exclusion: None,
      }),
      DynamicInputTag::ExclusiveFloat(prob, i) => Some(Self {
        prob: prob.clone(),
        external_tag: None,
        exclusion: Some(i.clone()),
      }),
      DynamicInputTag::Tensor(t) => Some(Self {
        prob: t.get_f64(),
        external_tag: T::from_tensor(t.clone()),
        exclusion: None,
      }),
    }
  }
}

impl<T: FromTensor> ConvertFromInputTag<()> for InputExclusiveDiffProb<T> {
  fn from_input_tag(_: ()) -> Option<Self> {
    None
  }
}

impl<T: FromTensor> ConvertFromInputTag<bool> for InputExclusiveDiffProb<T> {
  fn from_input_tag(b: bool) -> Option<Self> {
    if b {
      None
    } else {
      Some(Self::new_without_gradient(0.0, None))
    }
  }
}

impl<T: FromTensor> ConvertFromInputTag<usize> for InputExclusiveDiffProb<T> {
  fn from_input_tag(u: usize) -> Option<Self> {
    if u > 0 {
      None
    } else {
      Some(Self::new_without_gradient(0.0, None))
    }
  }
}

impl<T: FromTensor> ConvertFromInputTag<Exclusion> for InputExclusiveDiffProb<T> {
  fn from_input_tag(e: Exclusion) -> Option<Self> {
    match e {
      Exclusion::Independent => None,
      Exclusion::Exclusive(eid) => Some(Self::new_without_gradient(1.0, Some(eid))),
    }
  }
}

impl<T: FromTensor> ConvertFromInputTag<f64> for InputExclusiveDiffProb<T> {
  fn from_input_tag(t: f64) -> Option<Self> {
    Some(Self::new_without_gradient(t, None))
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputExclusiveProb> for InputExclusiveDiffProb<T> {
  fn from_input_tag(t: InputExclusiveProb) -> Option<Self> {
    Some(Self::new_without_gradient(t.prob.clone(), t.exclusion.clone()))
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputDiffProb<T>> for InputExclusiveDiffProb<T> {
  fn from_input_tag(t: InputDiffProb<T>) -> Option<Self> {
    Some(Self::new_without_gradient(t.0, None))
  }
}

impl<T: FromTensor> ConvertFromInputTag<InputExclusiveDiffProb<T>> for InputExclusiveDiffProb<T> {
  fn from_input_tag(t: InputExclusiveDiffProb<T>) -> Option<Self> {
    Some(t.clone())
  }
}
