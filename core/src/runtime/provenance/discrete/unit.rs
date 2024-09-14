use crate::common::foreign_aggregate::*;
use crate::common::foreign_aggregates::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;

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

  fn name(&self) -> String {
    format!("unit")
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
}

impl Aggregator<UnitProvenance> for CountAggregator {
  fn aggregate(
    &self,
    _p: &UnitProvenance,
    _env: &RuntimeEnvironment,
    elems: DynamicElements<UnitProvenance>,
  ) -> DynamicElements<UnitProvenance> {
    vec![DynamicElement::new(elems.len(), Unit)]
  }
}

impl Aggregator<UnitProvenance> for ExistsAggregator {
  fn aggregate(
    &self,
    _p: &UnitProvenance,
    _env: &RuntimeEnvironment,
    elems: DynamicElements<UnitProvenance>,
  ) -> DynamicElements<UnitProvenance> {
    vec![DynamicElement::new(!elems.is_empty(), Unit)]
  }
}

impl Aggregator<UnitProvenance> for MinMaxAggregator {
  fn aggregate(
    &self,
    p: &UnitProvenance,
    _env: &RuntimeEnvironment,
    batch: DynamicElements<UnitProvenance>,
  ) -> DynamicElements<UnitProvenance> {
    self.discrete_min_max(p, batch.iter_tuples())
  }
}

impl Aggregator<UnitProvenance> for SumProdAggregator {
  fn aggregate(
    &self,
    _p: &UnitProvenance,
    _env: &RuntimeEnvironment,
    batch: DynamicElements<UnitProvenance>,
  ) -> DynamicElements<UnitProvenance> {
    let res = self.perform_sum_prod(batch.iter_tuples());
    vec![DynamicElement::new(res, Unit)]
  }
}
