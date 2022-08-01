use super::*;

pub struct DynamicIntersectDataflow<'a, T: Tag> {
  pub d1: Box<DynamicDataflow<'a, T>>,
  pub d2: Box<DynamicDataflow<'a, T>>,
  pub ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for DynamicIntersectDataflow<'a, T> {
  fn clone(&self) -> Self {
    Self {
      d1: self.d1.clone(),
      d2: self.d2.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, T: Tag> DynamicIntersectDataflow<'a, T> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    let op = IntersectOp { ctx: self.ctx };
    DynamicBatches::binary(self.d1.iter_stable(), self.d2.iter_stable(), op.into())
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    let op = IntersectOp { ctx: self.ctx };
    DynamicBatches::chain(vec![
      DynamicBatches::binary(self.d1.iter_stable(), self.d2.iter_recent(), op.clone().into()),
      DynamicBatches::binary(self.d1.iter_recent(), self.d2.iter_stable(), op.clone().into()),
      DynamicBatches::binary(self.d1.iter_recent(), self.d2.iter_recent(), op.clone().into()),
    ])
  }
}

pub struct IntersectOp<'a, T: Tag> {
  ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for IntersectOp<'a, T> {
  fn clone(&self) -> Self {
    Self { ctx: self.ctx }
  }
}

impl<'a, T: Tag> From<IntersectOp<'a, T>> for BatchBinaryOp<'a, T> {
  fn from(op: IntersectOp<'a, T>) -> Self {
    Self::Intersect(op)
  }
}

impl<'a, T: Tag> IntersectOp<'a, T> {
  pub fn apply(&self, mut i1: DynamicBatch<'a, T>, mut i2: DynamicBatch<'a, T>) -> DynamicBatch<'a, T> {
    let i1_curr = i1.next();
    let i2_curr = i2.next();
    DynamicBatch::Intersect(DynamicIntersectBatch {
      i1: Box::new(i1),
      i1_curr,
      i2: Box::new(i2),
      i2_curr,
      ctx: self.ctx,
    })
  }
}

pub struct DynamicIntersectBatch<'a, T: Tag> {
  i1: Box<DynamicBatch<'a, T>>,
  i1_curr: Option<DynamicElement<T>>,
  i2: Box<DynamicBatch<'a, T>>,
  i2_curr: Option<DynamicElement<T>>,
  ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for DynamicIntersectBatch<'a, T> {
  fn clone(&self) -> Self {
    Self {
      i1: self.i1.clone(),
      i1_curr: self.i1_curr.clone(),
      i2: self.i2.clone(),
      i2_curr: self.i2_curr.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, T: Tag> Iterator for DynamicIntersectBatch<'a, T> {
  type Item = DynamicElement<T>;

  fn next(&mut self) -> Option<Self::Item> {
    use std::cmp::Ordering;
    loop {
      match (&self.i1_curr, &self.i2_curr) {
        (Some(i1_curr_elem), Some(i2_curr_elem)) => match i1_curr_elem.tuple.cmp(&i2_curr_elem.tuple) {
          Ordering::Less => {
            self.i1_curr = self.i1.search_ahead(|i1_next| i1_next < &i2_curr_elem.tuple);
          }
          Ordering::Equal => {
            let tag = self.ctx.mult(&i1_curr_elem.tag, &i2_curr_elem.tag);
            let result = DynamicElement::new(i1_curr_elem.tuple.clone(), tag);
            self.i1_curr = self.i1.next();
            self.i2_curr = self.i2.next();
            return Some(result);
          }
          Ordering::Greater => {
            self.i2_curr = self.i2.search_ahead(|i2_next| i2_next < &i1_curr_elem.tuple);
          }
        },
        _ => return None,
      }
    }
  }
}
