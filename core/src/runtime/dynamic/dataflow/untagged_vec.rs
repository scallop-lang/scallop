use crate::common::tuple::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct DynamicUntaggedVec<'a, Prov: Provenance> {
  pub ctx: &'a Prov,
  pub tuples: Vec<Tuple>,
}

impl<'a, Prov: Provenance> DynamicUntaggedVec<'a, Prov> {
  pub fn new(ctx: &'a Prov, tuples: Vec<Tuple>) -> Self {
    Self { ctx, tuples }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicUntaggedVec<'a, Prov> {
  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::single(UntaggedVecBatch::new(self.ctx, self.tuples.clone()))
  }
}

#[derive(Clone)]
pub struct UntaggedVecBatch<'a, Prov: Provenance> {
  prov: &'a Prov,
  iter: std::vec::IntoIter<Tuple>,
}

impl<'a, Prov: Provenance> UntaggedVecBatch<'a, Prov> {
  pub fn new(ctx: &'a Prov, tuples: Vec<Tuple>) -> Self {
    Self {
      prov: ctx,
      iter: tuples.into_iter(),
    }
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for UntaggedVecBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    self.iter.next().map(|t| DynamicElement::new(t, self.prov.one()))
  }
}
