use super::*;
use crate::runtime::dynamic::*;
use crate::runtime::statics::*;

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

  fn dynamic_count(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let count = batch
      .into_iter()
      .fold(0usize, |acc, e| if e.tag { acc + 1 } else { acc });
    vec![DynamicElement::new(count, self.one())]
  }

  fn static_count<T: StaticTupleTrait>(&self, batch: StaticElements<T, Self>) -> StaticElements<usize, Self> {
    let count = batch
      .into_iter()
      .fold(0usize, |acc, e| if e.tag { acc + 1 } else { acc });
    vec![StaticElement::new(count, self.one())]
  }

  fn dynamic_top_k(&self, k: usize, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    unweighted_aggregate_top_k_helper(batch, k)
  }

  fn static_top_k<T: StaticTupleTrait>(&self, k: usize, batch: StaticElements<T, Self>) -> StaticElements<T, Self> {
    unweighted_aggregate_top_k_helper(batch, k)
  }
}
