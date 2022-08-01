use super::*;
use crate::common::tuple::Tuple;

pub struct DynamicProductDataflow<'a, T: Tag> {
  pub d1: Box<DynamicDataflow<'a, T>>,
  pub d2: Box<DynamicDataflow<'a, T>>,
  pub ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for DynamicProductDataflow<'a, T> {
  fn clone(&self) -> Self {
    Self {
      d1: self.d1.clone(),
      d2: self.d2.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, T: Tag> DynamicProductDataflow<'a, T> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    let op = ProductOp { ctx: self.ctx };
    DynamicBatches::binary(self.d1.iter_stable(), self.d2.iter_stable(), op.into())
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    let op = ProductOp { ctx: self.ctx };
    DynamicBatches::chain(vec![
      DynamicBatches::binary(self.d1.iter_stable(), self.d2.iter_recent(), op.clone().into()),
      DynamicBatches::binary(self.d1.iter_recent(), self.d2.iter_stable(), op.clone().into()),
      DynamicBatches::binary(self.d1.iter_recent(), self.d2.iter_recent(), op.clone().into()),
    ])
  }
}

pub struct ProductOp<'a, T: Tag> {
  ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for ProductOp<'a, T> {
  fn clone(&self) -> Self {
    Self { ctx: self.ctx }
  }
}

impl<'a, T: Tag> From<ProductOp<'a, T>> for BatchBinaryOp<'a, T> {
  fn from(op: ProductOp<'a, T>) -> Self {
    Self::Product(op)
  }
}

impl<'a, T: Tag> ProductOp<'a, T> {
  pub fn apply(&self, mut i1: DynamicBatch<'a, T>, i2: DynamicBatch<'a, T>) -> DynamicBatch<'a, T> {
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

pub struct DynamicProductBatch<'a, T: Tag> {
  i1: Box<DynamicBatch<'a, T>>,
  i1_curr: Option<DynamicElement<T>>,
  i2_source: Box<DynamicBatch<'a, T>>,
  i2_clone: Box<DynamicBatch<'a, T>>,
  ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for DynamicProductBatch<'a, T> {
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

impl<'a, T: Tag> Iterator for DynamicProductBatch<'a, T> {
  type Item = DynamicElement<T>;

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
