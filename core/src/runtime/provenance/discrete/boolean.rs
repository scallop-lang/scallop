use crate::common::foreign_aggregate::*;
use crate::common::foreign_aggregates::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;

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

  fn tagging_fn(&self, ext_tag: Self::InputTag) -> Self::Tag {
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

impl Aggregator<BooleanProvenance> for CountAggregator {
  fn aggregate(
    &self,
    _p: &BooleanProvenance,
    _env: &RuntimeEnvironment,
    elems: DynamicElements<BooleanProvenance>,
  ) -> DynamicElements<BooleanProvenance> {
    let cnt = elems.iter().fold(0usize, |c, e| if e.tag { c + 1 } else { c });
    vec![DynamicElement::new(cnt, true)]
  }
}

impl Aggregator<BooleanProvenance> for ExistsAggregator {
  fn aggregate(
    &self,
    _p: &BooleanProvenance,
    _env: &RuntimeEnvironment,
    elems: DynamicElements<BooleanProvenance>,
  ) -> DynamicElements<BooleanProvenance> {
    let exist = elems.iter().any(|e| e.tag);
    vec![DynamicElement::new(exist, true)]
  }
}

impl Aggregator<BooleanProvenance> for MinMaxAggregator {
  fn aggregate(
    &self,
    p: &BooleanProvenance,
    _env: &RuntimeEnvironment,
    batch: DynamicElements<BooleanProvenance>,
  ) -> DynamicElements<BooleanProvenance> {
    let elems = batch.iter().filter_map(|e| if e.tag { Some(&e.tuple) } else { None });
    self.discrete_min_max(p, elems)
  }
}

impl Aggregator<BooleanProvenance> for SumProdAggregator {
  fn aggregate(
    &self,
    _p: &BooleanProvenance,
    _env: &RuntimeEnvironment,
    batch: DynamicElements<BooleanProvenance>,
  ) -> DynamicElements<BooleanProvenance> {
    let elems = batch.iter().filter_map(|e| if e.tag { Some(&e.tuple) } else { None });
    let res = self.perform_sum_prod(elems);
    vec![DynamicElement::new(res, true)]
  }
}
