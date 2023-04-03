use super::*;
use crate::runtime::dynamic::*;
use crate::runtime::statics::*;
use crate::common::input_tag::*;

pub type Natural = usize;

impl StaticInputTag for Natural {}

#[derive(Clone, Debug, Default)]
pub struct NaturalProvenance;

impl Provenance for NaturalProvenance {
  type Tag = Natural;

  type InputTag = usize;

  type OutputTag = usize;

  fn name() -> &'static str {
    "natural"
  }

  fn tagging_fn(&self, t: Self::InputTag) -> Self::Tag {
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

  fn dynamic_count(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let count = batch
      .into_iter()
      .fold(0usize, |acc, e| if e.tag > 0 { acc + 1 } else { acc });
    vec![DynamicElement::new(count, self.one())]
  }

  fn static_count<T: StaticTupleTrait>(&self, batch: StaticElements<T, Self>) -> StaticElements<usize, Self> {
    let count = batch
      .into_iter()
      .fold(0usize, |acc, e| if e.tag > 0 { acc + 1 } else { acc });
    vec![StaticElement::new(count, self.one())]
  }

  fn dynamic_top_k(&self, k: usize, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    unweighted_aggregate_top_k_helper(batch, k)
  }

  fn static_top_k<T: StaticTupleTrait>(&self, k: usize, batch: StaticElements<T, Self>) -> StaticElements<T, Self> {
    unweighted_aggregate_top_k_helper(batch, k)
  }
}
