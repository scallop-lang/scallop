use crate::common::tuple::*;

use super::*;

pub struct DynamicJoinIndexedVecDataflow<'a, Prov: Provenance> {
  pub d1: DynamicDataflow<'a, Prov>,
  pub indexed_vec: &'a DynamicIndexedVecCollection<Prov>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicJoinIndexedVecDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      d1: self.d1.clone(),
      indexed_vec: self.indexed_vec,
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicJoinIndexedVecDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    let op = QueryIndexedVecOp::new(self.indexed_vec, self.ctx);
    DynamicBatches::unary(self.d1.iter_stable(), op)
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    let op = QueryIndexedVecOp::new(self.indexed_vec, self.ctx);
    DynamicBatches::unary(self.d1.iter_recent(), op)
  }
}

pub struct QueryIndexedVecOp<'a, Prov: Provenance> {
  pub indexed_vec: &'a DynamicIndexedVecCollection<Prov>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> QueryIndexedVecOp<'a, Prov> {
  pub fn new(indexed_vec: &'a DynamicIndexedVecCollection<Prov>, ctx: &'a Prov) -> Self {
    Self { indexed_vec, ctx }
  }
}

impl<'a, Prov: Provenance> Clone for QueryIndexedVecOp<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      indexed_vec: self.indexed_vec,
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> BatchUnaryOp<'a, Prov> for QueryIndexedVecOp<'a, Prov> {
  fn apply(&self, b: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    DynamicBatch::new(DynamicJoinIndexedVecBatch {
      left: b,
      right: self.indexed_vec,
      ctx: self.ctx,
    })
  }
}

pub struct DynamicJoinIndexedVecBatch<'a, Prov: Provenance> {
  left: DynamicBatch<'a, Prov>,
  right: &'a DynamicIndexedVecCollection<Prov>,
  ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicJoinIndexedVecBatch<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      left: self.left.clone(),
      right: self.right,
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicJoinIndexedVecBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    while let Some(left_elem) = self.left.next_elem() {
      let DynamicElement { tuple, tag: left_tag } = left_elem;
      let key = &tuple[0];
      let index = key.as_usize();
      if let Some((tail, right_tag)) = self.right.get(index) {
        let merged_tag = self.ctx.mult(&left_tag, &right_tag);
        let merged_tup = Tuple::from((key.clone(), tuple[1].clone(), tail));
        return Some(DynamicElement::new(merged_tup, merged_tag))
      }
    }
    None
  }
}
