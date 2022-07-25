use super::utils::*;
use super::*;
use crate::common::tuple::Tuple;

pub struct DynamicJoinDataflow<'a, T: Tag> {
  pub d1: Box<DynamicDataflow<'a, T>>,
  pub d2: Box<DynamicDataflow<'a, T>>,
  pub ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for DynamicJoinDataflow<'a, T> {
  fn clone(&self) -> Self {
    Self {
      d1: self.d1.clone(),
      d2: self.d2.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, T: Tag> DynamicJoinDataflow<'a, T> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    let op = JoinOp { ctx: self.ctx };
    DynamicBatches::binary(self.d1.iter_stable(), self.d2.iter_stable(), op.into())
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    let op = JoinOp { ctx: self.ctx };
    DynamicBatches::chain(vec![
      DynamicBatches::binary(
        self.d1.iter_stable(),
        self.d2.iter_recent(),
        op.clone().into(),
      ),
      DynamicBatches::binary(
        self.d1.iter_recent(),
        self.d2.iter_stable(),
        op.clone().into(),
      ),
      DynamicBatches::binary(
        self.d1.iter_recent(),
        self.d2.iter_recent(),
        op.clone().into(),
      ),
    ])
  }
}

pub struct JoinOp<'a, T: Tag> {
  ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for JoinOp<'a, T> {
  fn clone(&self) -> Self {
    Self { ctx: self.ctx }
  }
}

impl<'a, T: Tag> From<JoinOp<'a, T>> for BatchBinaryOp<'a, T> {
  fn from(op: JoinOp<'a, T>) -> Self {
    Self::Join(op)
  }
}

impl<'a, T: Tag> JoinOp<'a, T> {
  pub fn apply(
    &self,
    mut i1: DynamicBatch<'a, T>,
    mut i2: DynamicBatch<'a, T>,
  ) -> DynamicBatch<'a, T> {
    let i1_curr = i1.next();
    let i2_curr = i2.next();
    DynamicBatch::Join(DynamicJoinBatch {
      i1: Box::new(i1),
      i1_curr,
      i2: Box::new(i2),
      i2_curr,
      curr_iter: None,
      ctx: self.ctx,
    })
  }
}

pub struct DynamicJoinBatch<'a, T: Tag> {
  i1: Box<DynamicBatch<'a, T>>,
  i1_curr: Option<DynamicElement<T>>,
  i2: Box<DynamicBatch<'a, T>>,
  i2_curr: Option<DynamicElement<T>>,
  curr_iter: Option<JoinProductIterator<T>>,
  ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for DynamicJoinBatch<'a, T> {
  fn clone(&self) -> Self {
    Self {
      i1: self.i1.clone(),
      i1_curr: self.i1_curr.clone(),
      i2: self.i2.clone(),
      i2_curr: self.i2_curr.clone(),
      curr_iter: self.curr_iter.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, T: Tag> Iterator for DynamicJoinBatch<'a, T> {
  type Item = DynamicElement<T>;

  fn next(&mut self) -> Option<Self::Item> {
    use std::cmp::Ordering;
    loop {
      if let Some(curr_prod_iter) = &mut self.curr_iter {
        if let Some((e1, e2)) = curr_prod_iter.next() {
          let tuple = Tuple::from((
            e1.tuple[0].clone(),
            e1.tuple[1].clone(),
            e2.tuple[1].clone(),
          ));
          let tag = self.ctx.mult(&e1.tag, &e2.tag);
          let result = DynamicElement::new(tuple, tag);
          return Some(result);
        } else {
          self.i1.step(curr_prod_iter.v1.len() - 1);
          self.i1_curr = self.i1.next();
          self.i2.step(curr_prod_iter.v2.len() - 1);
          self.i2_curr = self.i2.next();
          self.curr_iter = None;
        }
      }

      match (&self.i1_curr, &self.i2_curr) {
        (Some(i1_curr_elem), Some(i2_curr_elem)) => {
          match i1_curr_elem.tuple[0].cmp(&i2_curr_elem.tuple[0]) {
            Ordering::Less => {
              self.i1_curr = self
                .i1
                .search_ahead(|i1_next| i1_next[0] < i2_curr_elem.tuple[0])
            }
            Ordering::Equal => {
              let key = &i1_curr_elem.tuple[0];
              let v1 = std::iter::once(i1_curr_elem.clone())
                .chain(self.i1.clone().take_while(|x| &x.tuple[0] == key))
                .collect::<Vec<_>>();
              let v2 = std::iter::once(i2_curr_elem.clone())
                .chain(self.i2.clone().take_while(|x| &x.tuple[0] == key))
                .collect::<Vec<_>>();
              let iter = JoinProductIterator::new(v1, v2);
              self.curr_iter = Some(iter);
            }
            Ordering::Greater => {
              self.i2_curr = self
                .i2
                .search_ahead(|i2_next| i2_next[0] < i1_curr_elem.tuple[0])
            }
          }
        }
        _ => break None,
      }
    }
  }
}
