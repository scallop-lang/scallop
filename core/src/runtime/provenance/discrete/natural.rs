use super::*;

pub type Natural = usize;

#[derive(Clone, Debug, Default)]
pub struct NaturalProvenance;

impl Provenance for NaturalProvenance {
  type Tag = Natural;

  type InputTag = usize;

  type OutputTag = usize;

  fn name() -> &'static str {
    "natural"
  }

  fn tagging_fn(&mut self, t: Self::InputTag) -> Self::Tag {
    t
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    *t
  }

  fn discard(&self, t: &Self::Tag) -> bool {
    *t == 0
  }

  fn zero(&self) -> Self::Tag {
    0
  }

  fn one(&self) -> Self::Tag {
    1
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    t1 + t2
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    t1 * t2
  }

  fn saturated(&self, _: &Self::Tag, _: &Self::Tag) -> bool {
    true
  }
}
