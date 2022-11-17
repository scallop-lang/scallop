use super::*;

#[derive(Clone, Debug, Default)]
pub struct Unit;

impl std::fmt::Display for Unit {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("()")
  }
}

impl Tag for Unit {}

#[derive(Clone, Debug, Default)]
pub struct UnitProvenance;

impl Provenance for UnitProvenance {
  type Tag = Unit;

  type InputTag = ();

  type OutputTag = Unit;

  fn name() -> &'static str {
    "unit"
  }

  fn tagging_fn(&mut self, _: Self::InputTag) -> Self::Tag {
    Unit
  }

  fn recover_fn(&self, _: &Self::Tag) -> Self::OutputTag {
    Unit
  }

  fn discard(&self, _: &Self::Tag) -> bool {
    false
  }

  fn zero(&self) -> Self::Tag {
    Unit
  }

  fn one(&self) -> Self::Tag {
    Unit
  }

  fn add(&self, _: &Self::Tag, _: &Self::Tag) -> Self::Tag {
    Unit
  }

  fn mult(&self, _: &Self::Tag, _: &Self::Tag) -> Self::Tag {
    Unit
  }

  fn negate(&self, _: &Self::Tag) -> Option<Self::Tag> {
    None
  }

  fn saturated(&self, _: &Self::Tag, _: &Self::Tag) -> bool {
    true
  }
}
