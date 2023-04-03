use crate::common::input_tag::*;

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
      _ => None,
    }
  }
}
