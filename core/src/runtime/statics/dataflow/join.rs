use std::cmp::Ordering;
use std::marker::PhantomData;

use super::utils::*;
use super::*;
use crate::runtime::provenance::*;

pub fn join<'b, D1, D2, K, T1, T2, T>(
  d1: D1,
  d2: D2,
  semiring_ctx: &'b T::Context,
) -> Join<'b, D1, D2, K, T1, T2, T>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  D1: Dataflow<(K, T1), T>,
  D2: Dataflow<(K, T2), T>,
{
  Join {
    d1,
    d2,
    semiring_ctx,
    phantom: PhantomData,
  }
}

pub struct Join<'b, D1, D2, K, T1, T2, T>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  D1: Dataflow<(K, T1), T>,
  D2: Dataflow<(K, T2), T>,
{
  d1: D1,
  d2: D2,
  semiring_ctx: &'b T::Context,
  phantom: PhantomData<(K, T1, T2, T)>,
}

impl<'b, D1, D2, K, T1, T2, T> Clone for Join<'b, D1, D2, K, T1, T2, T>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  D1: Dataflow<(K, T1), T>,
  D2: Dataflow<(K, T2), T>,
{
  fn clone(&self) -> Self {
    Self {
      d1: self.d1.clone(),
      d2: self.d2.clone(),
      semiring_ctx: self.semiring_ctx,
      phantom: PhantomData,
    }
  }
}

impl<'b, D1, D2, K, T1, T2, T> Dataflow<(K, T1, T2), T> for Join<'b, D1, D2, K, T1, T2, T>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  D1: Dataflow<(K, T1), T>,
  D2: Dataflow<(K, T2), T>,
{
  type Stable = BatchesJoin<
    D1::Stable,
    D2::Stable,
    StableStableOp<'b, D1, D2, K, T1, T2, T>,
    (K, T1),
    (K, T2),
    (K, T1, T2),
    T,
  >;

  type Recent = BatchesChain3<
    BatchesJoin<
      D1::Recent,
      D2::Stable,
      RecentStableOp<'b, D1, D2, K, T1, T2, T>,
      (K, T1),
      (K, T2),
      (K, T1, T2),
      T,
    >,
    BatchesJoin<
      D1::Stable,
      D2::Recent,
      StableRecentOp<'b, D1, D2, K, T1, T2, T>,
      (K, T1),
      (K, T2),
      (K, T1, T2),
      T,
    >,
    BatchesJoin<
      D1::Recent,
      D2::Recent,
      RecentRecentOp<'b, D1, D2, K, T1, T2, T>,
      (K, T1),
      (K, T2),
      (K, T1, T2),
      T,
    >,
    (K, T1, T2),
    T,
  >;

  fn iter_stable(&self) -> Self::Stable {
    let op = JoinOp::new(self.semiring_ctx);
    Self::Stable::join(self.d1.iter_stable(), self.d2.iter_stable(), op)
  }

  fn iter_recent(self) -> Self::Recent {
    let d1_stable = self.d1.iter_stable();
    let d2_stable = self.d2.iter_stable();
    let d1_recent = self.d1.iter_recent();
    let d2_recent = self.d2.iter_recent();
    Self::Recent::chain_3(
      BatchesJoin::join(d1_recent.clone(), d2_stable, JoinOp::new(self.semiring_ctx)),
      BatchesJoin::join(d1_stable, d2_recent.clone(), JoinOp::new(self.semiring_ctx)),
      BatchesJoin::join(d1_recent, d2_recent, JoinOp::new(self.semiring_ctx)),
    )
  }
}

type StableStableOp<'b, D1, D2, K, T1, T2, T> = JoinOp<
  'b,
  <<D1 as Dataflow<(K, T1), T>>::Stable as Batches<(K, T1), T>>::Batch,
  <<D2 as Dataflow<(K, T2), T>>::Stable as Batches<(K, T2), T>>::Batch,
  K,
  T1,
  T2,
  T,
>;

type RecentStableOp<'b, D1, D2, K, T1, T2, T> = JoinOp<
  'b,
  <<D1 as Dataflow<(K, T1), T>>::Recent as Batches<(K, T1), T>>::Batch,
  <<D2 as Dataflow<(K, T2), T>>::Stable as Batches<(K, T2), T>>::Batch,
  K,
  T1,
  T2,
  T,
>;

type StableRecentOp<'b, D1, D2, K, T1, T2, T> = JoinOp<
  'b,
  <<D1 as Dataflow<(K, T1), T>>::Stable as Batches<(K, T1), T>>::Batch,
  <<D2 as Dataflow<(K, T2), T>>::Recent as Batches<(K, T2), T>>::Batch,
  K,
  T1,
  T2,
  T,
>;

type RecentRecentOp<'b, D1, D2, K, T1, T2, T> = JoinOp<
  'b,
  <<D1 as Dataflow<(K, T1), T>>::Recent as Batches<(K, T1), T>>::Batch,
  <<D2 as Dataflow<(K, T2), T>>::Recent as Batches<(K, T2), T>>::Batch,
  K,
  T1,
  T2,
  T,
>;

pub struct JoinOp<'b, I1, I2, K, T1, T2, T>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<(K, T1), T>,
  I2: Batch<(K, T2), T>,
{
  semiring_ctx: &'b T::Context,
  phantom: PhantomData<(I1, I2, K, T1, T2, T)>,
}

impl<'b, I1, I2, K, T1, T2, T> Clone for JoinOp<'b, I1, I2, K, T1, T2, T>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<(K, T1), T>,
  I2: Batch<(K, T2), T>,
{
  fn clone(&self) -> Self {
    Self {
      semiring_ctx: self.semiring_ctx,
      phantom: PhantomData,
    }
  }
}

impl<'b, I1, I2, K, T1, T2, T> JoinOp<'b, I1, I2, K, T1, T2, T>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<(K, T1), T>,
  I2: Batch<(K, T2), T>,
{
  pub fn new(semiring_ctx: &'b T::Context) -> Self {
    Self {
      semiring_ctx,
      phantom: PhantomData,
    }
  }
}

impl<'b, I1, I2, K, T1, T2, T> BatchBinaryOp<I1, I2> for JoinOp<'b, I1, I2, K, T1, T2, T>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<(K, T1), T>,
  I2: Batch<(K, T2), T>,
{
  type IOut = JoinIterator<'b, I1, I2, K, T1, T2, T>;

  fn apply(&self, mut i1: I1, mut i2: I2) -> Self::IOut {
    let i1_curr = i1.next();
    let i2_curr = i2.next();
    Self::IOut {
      i1,
      i2,
      i1_curr,
      i2_curr,
      curr_iter: None,
      semiring_ctx: self.semiring_ctx,
    }
  }
}

pub struct JoinIterator<'b, I1, I2, K, T1, T2, T>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<(K, T1), T>,
  I2: Batch<(K, T2), T>,
{
  i1: I1,
  i2: I2,
  i1_curr: Option<StaticElement<(K, T1), T>>,
  i2_curr: Option<StaticElement<(K, T2), T>>,
  curr_iter: Option<JoinProductIterator<(K, T1), (K, T2), T>>,
  semiring_ctx: &'b T::Context,
}

impl<'b, I1, I2, K, T1, T2, T> Clone for JoinIterator<'b, I1, I2, K, T1, T2, T>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<(K, T1), T>,
  I2: Batch<(K, T2), T>,
{
  fn clone(&self) -> Self {
    Self {
      i1: self.i1.clone(),
      i2: self.i2.clone(),
      i1_curr: self.i1_curr.clone(),
      i2_curr: self.i2_curr.clone(),
      curr_iter: self.curr_iter.clone(),
      semiring_ctx: self.semiring_ctx,
    }
  }
}

impl<'b, I1, I2, K, T1, T2, T> Iterator for JoinIterator<'b, I1, I2, K, T1, T2, T>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<(K, T1), T>,
  I2: Batch<(K, T2), T>,
{
  type Item = StaticElement<(K, T1, T2), T>;

  fn next(&mut self) -> Option<Self::Item> {
    loop {
      // First go through curr joint product iterator
      if let Some(curr_iter) = &mut self.curr_iter {
        if let Some((e1, e2)) = curr_iter.next() {
          let (k, t1) = e1.tuple.0;
          let (_, t2) = e2.tuple.0;
          let tup = (k, t1, t2);
          let tag = self.semiring_ctx.mult(&e1.tag, &e2.tag);
          let result = StaticElement::new(tup, tag);
          return Some(result);
        } else {
          // Skip ahead
          self.i1.step(curr_iter.v1.len() - 1);
          self.i1_curr = self.i1.next();
          self.i2.step(curr_iter.v2.len() - 1);
          self.i2_curr = self.i2.next();

          // Remove current iterator
          self.curr_iter = None;
        }
      }

      // Then continue
      match (&self.i1_curr, &self.i2_curr) {
        (Some(i1_curr), Some(i2_curr)) => {
          match i1_curr.tuple.0 .0.partial_cmp(&i2_curr.tuple.0 .0).unwrap() {
            Ordering::Less => {
              self.i1_curr = self
                .i1
                .search_ahead(|i1_next| i1_next.0 .0 < i2_curr.tuple.0 .0)
            }
            Ordering::Equal => {
              let key = &i1_curr.tuple.0 .0;
              let v1 = std::iter::once(i1_curr.clone())
                .chain(self.i1.clone().take_while(|x| &x.tuple.0 .0 == key))
                .collect::<Vec<_>>();
              let v2 = std::iter::once(i2_curr.clone())
                .chain(self.i2.clone().take_while(|x| &x.tuple.0 .0 == key))
                .collect::<Vec<_>>();
              let iter = JoinProductIterator::new(v1, v2);
              self.curr_iter = Some(iter);
            }
            Ordering::Greater => {
              self.i2_curr = self
                .i2
                .search_ahead(|i2_next| i2_next.0 .0 < i1_curr.tuple.0 .0)
            }
          }
        }
        _ => break None,
      }
    }
  }
}

impl<'b, I1, I2, K, T1, T2, T> Batch<(K, T1, T2), T> for JoinIterator<'b, I1, I2, K, T1, T2, T>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<(K, T1), T>,
  I2: Batch<(K, T2), T>,
{
}
