use super::*;

#[derive(Clone, Debug, Default)]
pub struct RealTropicalProvenance;

impl Provenance for RealTropicalProvenance {
  type Tag = f64;

  type InputTag = f64;

  type OutputTag = f64;

  fn name(&self) -> String {
    format!("realtropical")
  }

  fn tagging_fn(&self, ext_tag: Self::InputTag) -> Self::Tag {
    ext_tag.into()
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    t.clone()
  }

  fn discard(&self, t: &Self::Tag) -> bool {
    t.is_infinite()
  }

  fn weight(&self, tag: &Self::Tag) -> f64 {
    (-tag).exp()
  }

  fn zero(&self) -> Self::Tag {
    std::f64::INFINITY
  }

  fn one(&self) -> Self::Tag {
    0.0
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    t1.min(*t2)
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    t1 + t2
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    (t_old - t_new).abs() < 0.001
  }
}
