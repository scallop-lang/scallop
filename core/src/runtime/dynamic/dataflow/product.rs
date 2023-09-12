use super::*;
use crate::common::tuple::Tuple;

pub struct DynamicProductDataflow<'a, Prov: Provenance> {
  pub d1: DynamicDataflow<'a, Prov>,
  pub d2: DynamicDataflow<'a, Prov>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicProductDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      d1: self.d1.clone(),
      d2: self.d2.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicProductDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    let op = ProductOp { ctx: self.ctx };
    DynamicBatches::binary(self.d1.iter_stable(), self.d2.iter_stable(), op)
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    let op = ProductOp { ctx: self.ctx };
    DynamicBatches::chain(vec![
      DynamicBatches::binary(self.d1.iter_stable(), self.d2.iter_recent(), op.clone()),
      DynamicBatches::binary(self.d1.iter_recent(), self.d2.iter_stable(), op.clone()),
      DynamicBatches::binary(self.d1.iter_recent(), self.d2.iter_recent(), op.clone()),
    ])
  }
}

pub struct ProductOp<'a, Prov: Provenance> {
  ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for ProductOp<'a, Prov> {
  fn clone(&self) -> Self {
    Self { ctx: self.ctx }
  }
}

impl<'a, Prov: Provenance> BatchBinaryOp<'a, Prov> for ProductOp<'a, Prov> {
  fn apply(&self, mut i1: DynamicBatch<'a, Prov>, i2: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    let i1_curr = i1.next_elem();
    DynamicBatch::new(DynamicProductBatch {
      i1: i1,
      i1_curr,
      i2_source: i2.clone(),
      i2_clone: i2,
      ctx: self.ctx,
    })
  }
}

pub struct DynamicProductBatch<'a, Prov: Provenance> {
  i1: DynamicBatch<'a, Prov>,
  i1_curr: Option<DynamicElement<Prov>>,
  i2_source: DynamicBatch<'a, Prov>,
  i2_clone: DynamicBatch<'a, Prov>,
  ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicProductBatch<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      i1: self.i1.clone(),
      i1_curr: self.i1_curr.clone(),
      i2_source: self.i2_source.clone(),
      i2_clone: self.i2_clone.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicProductBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    while let Some(i1_elem) = &self.i1_curr {
      match self.i2_clone.next_elem() {
        Some(i2_elem) => {
          let tuple = Tuple::from((i1_elem.tuple.clone(), i2_elem.tuple.clone()));
          let tag = self.ctx.mult(&i1_elem.tag, &i2_elem.tag);
          let elem = DynamicElement::new(tuple, tag);
          return Some(elem);
        }
        None => {
          self.i1_curr = self.i1.next_elem();
          self.i2_clone = self.i2_source.clone();
        }
      }
    }
    None
  }
}
