use crate::common::input_tag::*;

#[derive(Clone)]
pub struct InputExclusiveDiffProb<T: Clone + 'static> {
  /// The probability of the tag
  pub prob: f64,

  /// The external tag for differentiability
  pub external_tag: Option<T>,

  /// An optional identifier of the mutual exclusion
  pub exclusion: Option<usize>,
}

impl<T: Clone + 'static> InputExclusiveDiffProb<T> {
  pub fn new(prob: f64, tag: T, exclusion: Option<usize>) -> Self {
    Self { prob, external_tag: Some(tag), exclusion }
  }
}

impl<T: Clone + 'static> std::fmt::Debug for InputExclusiveDiffProb<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.prob.fmt(f)
  }
}

impl<T: Clone + 'static> From<(f64, T, Option<usize>)> for InputExclusiveDiffProb<T> {
  fn from((prob, tag, exclusion): (f64, T, Option<usize>)) -> Self {
    Self { prob, external_tag: Some(tag), exclusion }
  }
}

impl<T: Clone + 'static> StaticInputTag for InputExclusiveDiffProb<T> {
  fn from_dynamic_input_tag(t: &DynamicInputTag) -> Option<Self> {
    match t {
      DynamicInputTag::None => None,
      DynamicInputTag::Bool(b) => Some(Self { prob: if *b { 1.0 } else { 0.0 }, external_tag: None, exclusion: None }),
      DynamicInputTag::Exclusive(i) => Some(Self { prob: 1.0, external_tag: None, exclusion: Some(i.clone()) }),
      DynamicInputTag::Float(prob) => Some(Self { prob: prob.clone(), external_tag: None, exclusion: None }),
      DynamicInputTag::ExclusiveFloat(prob, i) => Some(Self { prob: prob.clone(), external_tag: None, exclusion: Some(i.clone()) }),
    }
  }
}
