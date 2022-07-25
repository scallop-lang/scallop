use super::*;

#[derive(Clone, Debug, Default)]
pub struct Unit;

impl std::fmt::Display for Unit {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("()")
  }
}

impl Tag for Unit {
  type Context = UnitContext;
}

#[derive(Clone, Debug, Default)]
pub struct UnitContext;

impl ProvenanceContext for UnitContext {
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
}
