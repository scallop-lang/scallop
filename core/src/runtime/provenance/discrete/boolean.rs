use super::*;

pub type Boolean = bool;

#[derive(Debug, Clone, Default)]
pub struct BooleanProvenance;

impl Provenance for BooleanProvenance {
  type Tag = Boolean;

  type InputTag = bool;

  type OutputTag = bool;

  fn name() -> &'static str {
    "boolean"
  }

  fn tagging_fn(&mut self, ext_tag: Self::InputTag) -> Self::Tag {
    ext_tag
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    *t
  }

  fn discard(&self, t: &Self::Tag) -> bool {
    !t
  }

  fn zero(&self) -> Self::Tag {
    false
  }

  fn one(&self) -> Self::Tag {
    true
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    *t1 || *t2
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    *t1 && *t2
  }

  fn minus(&self, t1: &Self::Tag, t2: &Self::Tag) -> Option<Self::Tag> {
    Some(*t1 && !t2)
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    t_old == t_new
  }
}
