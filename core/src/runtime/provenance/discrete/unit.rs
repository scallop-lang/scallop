use crate::runtime::dynamic::*;
use crate::runtime::statics::*;

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

impl UnitProvenance {
  pub fn new() -> Self {
    Self
  }
}

impl Provenance for UnitProvenance {
  type Tag = Unit;

  type InputTag = ();

  type OutputTag = Unit;

  fn name() -> &'static str {
    "unit"
  }

  fn tagging_fn(&self, _: Self::InputTag) -> Self::Tag {
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

  fn dynamic_top_k(&self, k: usize, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    unweighted_aggregate_top_k_helper(batch, k)
  }

  fn static_top_k<T: StaticTupleTrait>(&self, k: usize, batch: StaticElements<T, Self>) -> StaticElements<T, Self> {
    unweighted_aggregate_top_k_helper(batch, k)
  }
}
