use super::*;

#[derive(Clone, Debug)]
pub struct AddMultProbProvenance {
  valid_threshold: f64,
}

impl AddMultProbProvenance {
  /// The soft comparison between two probabilities
  ///
  /// This function is commonly used for testing purpose
  pub fn soft_cmp(fst: &f64, snd: &f64) -> bool {
    (fst - snd).abs() < 0.001
  }
}

impl Default for AddMultProbProvenance {
  fn default() -> Self {
    Self {
      valid_threshold: 0.0000,
    }
  }
}

impl Provenance for AddMultProbProvenance {
  type Tag = f64;

  type InputTag = f64;

  type OutputTag = f64;

  fn name() -> &'static str {
    "addmultprob"
  }

  fn tagging_fn(&self, p: Self::InputTag) -> Self::Tag {
    p.into()
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    *t
  }

  fn discard(&self, p: &Self::Tag) -> bool {
    p <= &self.valid_threshold
  }

  fn zero(&self) -> Self::Tag {
    0.0
  }

  fn one(&self) -> Self::Tag {
    1.0
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    (t1 + t2).min(1.0)
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    t1 * t2
  }

  fn negate(&self, p: &Self::Tag) -> Option<Self::Tag> {
    Some(1.0 - p)
  }

  fn saturated(&self, _: &Self::Tag, _: &Self::Tag) -> bool {
    true
  }

  fn weight(&self, t: &Self::Tag) -> f64 {
    *t as f64
  }
}
