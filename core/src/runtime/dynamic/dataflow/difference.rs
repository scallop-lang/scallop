use super::*;

pub struct DynamicDifferenceDataflow<'a, T: Tag> {
  pub d1: Box<DynamicDataflow<'a, T>>,
  pub d2: Box<DynamicDataflow<'a, T>>,
  pub ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for DynamicDifferenceDataflow<'a, T> {
  fn clone(&self) -> Self {
    Self {
      d1: self.d1.clone(),
      d2: self.d2.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, T: Tag> DynamicDifferenceDataflow<'a, T> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::Empty
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    let op = DifferenceOp { ctx: self.ctx };
    DynamicBatches::chain(vec![
      DynamicBatches::binary(self.d1.iter_stable(), self.d2.iter_recent(), op.clone().into()),
      DynamicBatches::binary(self.d1.iter_recent(), self.d2.iter_stable(), op.clone().into()),
      DynamicBatches::binary(self.d1.iter_recent(), self.d2.iter_recent(), op.clone().into()),
    ])
  }
}

pub struct DifferenceOp<'a, T: Tag> {
  ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for DifferenceOp<'a, T> {
  fn clone(&self) -> Self {
    Self { ctx: self.ctx }
  }
}

impl<'a, T: Tag> From<DifferenceOp<'a, T>> for BatchBinaryOp<'a, T> {
  fn from(op: DifferenceOp<'a, T>) -> Self {
    Self::Difference(op)
  }
}

impl<'a, T: Tag> DifferenceOp<'a, T> {
  pub fn apply(&self, mut i1: DynamicBatch<'a, T>, mut i2: DynamicBatch<'a, T>) -> DynamicBatch<'a, T> {
    let i1_curr = i1.next();
    let i2_curr = i2.next();
    DynamicBatch::Difference(DynamicDifferenceBatch {
      i1: Box::new(i1),
      i1_curr,
      i2: Box::new(i2),
      i2_curr,
      ctx: self.ctx,
    })
  }
}

pub struct DynamicDifferenceBatch<'a, T: Tag> {
  i1: Box<DynamicBatch<'a, T>>,
  i1_curr: Option<DynamicElement<T>>,
  i2: Box<DynamicBatch<'a, T>>,
  i2_curr: Option<DynamicElement<T>>,
  ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for DynamicDifferenceBatch<'a, T> {
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

impl<'a, T: Tag> Iterator for DynamicDifferenceBatch<'a, T> {
  type Item = DynamicElement<T>;

  fn next(&mut self) -> Option<Self::Item> {
    use std::cmp::Ordering;
    loop {
      match (&self.i1_curr, &self.i2_curr) {
        (Some(i1_curr_elem), Some(i2_curr_elem)) => match i1_curr_elem.tuple.cmp(&i2_curr_elem.tuple) {
          Ordering::Less => {
            let result = i1_curr_elem.clone();
            self.i1_curr = self.i1.next();
            return Some(result);
          }
          Ordering::Equal => {
            let maybe_tag = self.ctx.minus(&i1_curr_elem.tag, &i2_curr_elem.tag);
            if let Some(tag) = maybe_tag {
              let result = DynamicElement::new(i1_curr_elem.tuple.clone(), tag);
              self.i1_curr = self.i1.next();
              self.i2_curr = self.i2.next();
              return Some(result);
            } else {
              self.i1_curr = self.i1.next();
              self.i2_curr = self.i2.next();
            }
          }
          Ordering::Greater => {
            self.i2_curr = self.i2.search_ahead(|i2_next| i2_next < &i1_curr_elem.tuple);
          }
        },
        (Some(i1_curr_elem), None) => {
          let result = i1_curr_elem.clone();
          self.i1_curr = self.i1.next();
          return Some(result);
        }
        _ => return None,
      }
    }
  }
}
