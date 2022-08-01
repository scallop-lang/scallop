use std::cmp::Ordering;
use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;

pub fn intersect<'b, D1, D2, Tup, T>(d1: D1, d2: D2, semiring_ctx: &'b T::Context) -> Intersection<'b, D1, D2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  D1: Dataflow<Tup, T>,
  D2: Dataflow<Tup, T>,
{
  Intersection {
    d1,
    d2,
    semiring_ctx,
    phantom: PhantomData,
  }
}

pub struct Intersection<'b, D1, D2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  D1: Dataflow<Tup, T>,
  D2: Dataflow<Tup, T>,
{
  d1: D1,
  d2: D2,
  semiring_ctx: &'b T::Context,
  phantom: PhantomData<(Tup, T)>,
}

impl<'b, D1, D2, Tup, T> Clone for Intersection<'b, D1, D2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  D1: Dataflow<Tup, T>,
  D2: Dataflow<Tup, T>,
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

impl<'b, D1, D2, Tup, T> Dataflow<Tup, T> for Intersection<'b, D1, D2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  D1: Dataflow<Tup, T>,
  D2: Dataflow<Tup, T>,
{
  type Stable = BatchesJoin<D1::Stable, D2::Stable, StableStableOp<'b, D1, D2, Tup, T>, Tup, Tup, Tup, T>;

  type Recent = BatchesChain3<
    BatchesJoin<D1::Recent, D2::Stable, RecentStableOp<'b, D1, D2, Tup, T>, Tup, Tup, Tup, T>,
    BatchesJoin<D1::Stable, D2::Recent, StableRecentOp<'b, D1, D2, Tup, T>, Tup, Tup, Tup, T>,
    BatchesJoin<D1::Recent, D2::Recent, RecentRecentOp<'b, D1, D2, Tup, T>, Tup, Tup, Tup, T>,
    Tup,
    T,
  >;

  fn iter_stable(&self) -> Self::Stable {
    let op = IntersectionOp::new(self.semiring_ctx);
    Self::Stable::join(self.d1.iter_stable(), self.d2.iter_stable(), op)
  }

  fn iter_recent(self) -> Self::Recent {
    let d1_stable = self.d1.iter_stable();
    let d2_stable = self.d2.iter_stable();
    let d1_recent = self.d1.iter_recent();
    let d2_recent = self.d2.iter_recent();
    Self::Recent::chain_3(
      BatchesJoin::join(d1_recent.clone(), d2_stable, IntersectionOp::new(self.semiring_ctx)),
      BatchesJoin::join(d1_stable, d2_recent.clone(), IntersectionOp::new(self.semiring_ctx)),
      BatchesJoin::join(d1_recent, d2_recent, IntersectionOp::new(self.semiring_ctx)),
    )
  }
}

type StableStableOp<'b, D1, D2, Tup, T> = IntersectionOp<
  'b,
  <<D1 as Dataflow<Tup, T>>::Stable as Batches<Tup, T>>::Batch,
  <<D2 as Dataflow<Tup, T>>::Stable as Batches<Tup, T>>::Batch,
  Tup,
  T,
>;

type RecentStableOp<'b, D1, D2, Tup, T> = IntersectionOp<
  'b,
  <<D1 as Dataflow<Tup, T>>::Recent as Batches<Tup, T>>::Batch,
  <<D2 as Dataflow<Tup, T>>::Stable as Batches<Tup, T>>::Batch,
  Tup,
  T,
>;

type StableRecentOp<'b, D1, D2, Tup, T> = IntersectionOp<
  'b,
  <<D1 as Dataflow<Tup, T>>::Stable as Batches<Tup, T>>::Batch,
  <<D2 as Dataflow<Tup, T>>::Recent as Batches<Tup, T>>::Batch,
  Tup,
  T,
>;

type RecentRecentOp<'b, D1, D2, Tup, T> = IntersectionOp<
  'b,
  <<D1 as Dataflow<Tup, T>>::Recent as Batches<Tup, T>>::Batch,
  <<D2 as Dataflow<Tup, T>>::Recent as Batches<Tup, T>>::Batch,
  Tup,
  T,
>;

pub struct IntersectionOp<'a, I1, I2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  I1: Batch<Tup, T>,
  I2: Batch<Tup, T>,
{
  semiring_ctx: &'a T::Context,
  phantom: PhantomData<(I1, I2, Tup, T)>,
}

impl<'a, I1, I2, Tup, T> Clone for IntersectionOp<'a, I1, I2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  I1: Batch<Tup, T>,
  I2: Batch<Tup, T>,
{
  fn clone(&self) -> Self {
    Self {
      semiring_ctx: self.semiring_ctx,
      phantom: PhantomData,
    }
  }
}

impl<'a, I1, I2, Tup, T> IntersectionOp<'a, I1, I2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  I1: Batch<Tup, T>,
  I2: Batch<Tup, T>,
{
  pub fn new(semiring_ctx: &'a T::Context) -> Self {
    Self {
      semiring_ctx,
      phantom: PhantomData,
    }
  }
}

impl<'a, I1, I2, Tup, T> BatchBinaryOp<I1, I2> for IntersectionOp<'a, I1, I2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  I1: Batch<Tup, T>,
  I2: Batch<Tup, T>,
{
  type IOut = IntersectionIterator<'a, I1, I2, Tup, T>;

  fn apply(&self, mut i1: I1, mut i2: I2) -> Self::IOut {
    let i1_curr = i1.next();
    let i2_curr = i2.next();
    IntersectionIterator {
      i1,
      i2,
      i1_curr,
      i2_curr,
      semiring_ctx: self.semiring_ctx,
      phantom: PhantomData,
    }
  }
}

pub struct IntersectionIterator<'b, I1, I2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  I1: Batch<Tup, T>,
  I2: Batch<Tup, T>,
{
  i1: I1,
  i2: I2,
  i1_curr: Option<StaticElement<Tup, T>>,
  i2_curr: Option<StaticElement<Tup, T>>,
  semiring_ctx: &'b T::Context,
  phantom: PhantomData<(I1, I2, Tup)>,
}

impl<'b, I1, I2, Tup, T> Clone for IntersectionIterator<'b, I1, I2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  I1: Batch<Tup, T>,
  I2: Batch<Tup, T>,
{
  fn clone(&self) -> Self {
    Self {
      i1: self.i1.clone(),
      i2: self.i2.clone(),
      i1_curr: self.i1_curr.clone(),
      i2_curr: self.i2_curr.clone(),
      semiring_ctx: self.semiring_ctx,
      phantom: PhantomData,
    }
  }
}

impl<'b, I1, I2, Tup, T> Iterator for IntersectionIterator<'b, I1, I2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  I1: Batch<Tup, T>,
  I2: Batch<Tup, T>,
{
  type Item = StaticElement<Tup, T>;

  fn next(&mut self) -> Option<Self::Item> {
    loop {
      match (&self.i1_curr, &self.i2_curr) {
        (Some(i1_curr), Some(i2_curr)) => match i1_curr.tuple.cmp(&i2_curr.tuple) {
          Ordering::Less => {
            self.i1_curr = self.i1.search_ahead(|i1_next| i1_next < &i2_curr.tuple);
          }
          Ordering::Equal => {
            let tag = self.semiring_ctx.mult(&i1_curr.tag, &i2_curr.tag);
            let result = StaticElement::new(i1_curr.tuple.get().clone(), tag);
            self.i1_curr = self.i1.next();
            self.i2_curr = self.i2.next();
            return Some(result);
          }
          Ordering::Greater => {
            self.i2_curr = self.i2.search_ahead(|i2_next| i2_next < &i1_curr.tuple);
          }
        },
        _ => return None,
      }
    }
  }
}

impl<'b, I1, I2, Tup, T> Batch<Tup, T> for IntersectionIterator<'b, I1, I2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  I1: Batch<Tup, T>,
  I2: Batch<Tup, T>,
{
}
