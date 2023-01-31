use super::*;
use crate::common::tuple::Tuple;

pub struct DynamicProductDataflow<'a, Prov: Provenance> {
  pub d1: Box<DynamicDataflow<'a, Prov>>,
  pub d2: Box<DynamicDataflow<'a, Prov>>,
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

impl<'a, Prov: Provenance> DynamicProductDataflow<'a, Prov> {
  pub fn iter_stable(&self, runtime: &'a RuntimeEnvironment) -> DynamicBatches<'a, Prov> {
    let op = ProductOp { ctx: self.ctx };
    DynamicBatches::binary(self.d1.iter_stable(runtime), self.d2.iter_stable(runtime), op.into())
  }

  pub fn iter_recent(&self, runtime: &'a RuntimeEnvironment) -> DynamicBatches<'a, Prov> {
    let op = ProductOp { ctx: self.ctx };
    DynamicBatches::chain(vec![
      DynamicBatches::binary(
        self.d1.iter_stable(runtime),
        self.d2.iter_recent(runtime),
        op.clone().into(),
      ),
      DynamicBatches::binary(
        self.d1.iter_recent(runtime),
        self.d2.iter_stable(runtime),
        op.clone().into(),
      ),
      DynamicBatches::binary(
        self.d1.iter_recent(runtime),
        self.d2.iter_recent(runtime),
        op.clone().into(),
      ),
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

impl<'a, Prov: Provenance> From<ProductOp<'a, Prov>> for BatchBinaryOp<'a, Prov> {
  fn from(op: ProductOp<'a, Prov>) -> Self {
    Self::Product(op)
  }
}

impl<'a, Prov: Provenance> ProductOp<'a, Prov> {
  pub fn apply(&self, mut i1: DynamicBatch<'a, Prov>, i2: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    let i1_curr = i1.next();
    DynamicBatch::Product(DynamicProductBatch {
      i1: Box::new(i1),
      i1_curr,
      i2_source: Box::new(i2.clone()),
      i2_clone: Box::new(i2),
      ctx: self.ctx,
    })
  }
}

pub struct DynamicProductBatch<'a, Prov: Provenance> {
  i1: Box<DynamicBatch<'a, Prov>>,
  i1_curr: Option<DynamicElement<Prov>>,
  i2_source: Box<DynamicBatch<'a, Prov>>,
  i2_clone: Box<DynamicBatch<'a, Prov>>,
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

impl<'a, Prov: Provenance> Iterator for DynamicProductBatch<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    loop {
      match &self.i1_curr {
        Some(i1_elem) => match self.i2_clone.next() {
          Some(i2_elem) => {
            let tuple = Tuple::from((i1_elem.tuple.clone(), i2_elem.tuple.clone()));
            let tag = self.ctx.mult(&i1_elem.tag, &i2_elem.tag);
            let elem = DynamicElement::new(tuple, tag);
            return Some(elem);
          }
          None => {
            self.i1_curr = self.i1.next();
            self.i2_clone = self.i2_source.clone();
          }
        },
        None => return None,
      }
    }
  }
}
