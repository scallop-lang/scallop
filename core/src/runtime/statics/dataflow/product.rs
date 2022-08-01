use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;

pub fn product<'b, D1, D2, T1, T2, T>(d1: D1, d2: D2, semiring_ctx: &'b T::Context) -> Product<'b, D1, D2, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  D1: Dataflow<T1, T>,
  D2: Dataflow<T2, T>,
{
  Product {
    d1,
    d2,
    semiring_ctx,
    phantom: PhantomData,
  }
}

pub struct Product<'b, D1, D2, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  D1: Dataflow<T1, T>,
  D2: Dataflow<T2, T>,
{
  d1: D1,
  d2: D2,
  semiring_ctx: &'b T::Context,
  phantom: PhantomData<(T1, T2, T)>,
}

impl<'b, D1, D2, T1, T2, T> Clone for Product<'b, D1, D2, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  D1: Dataflow<T1, T>,
  D2: Dataflow<T2, T>,
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

impl<'b, D1, D2, T1, T2, T> Dataflow<(T1, T2), T> for Product<'b, D1, D2, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  D1: Dataflow<T1, T>,
  D2: Dataflow<T2, T>,
{
  type Stable = BatchesJoin<D1::Stable, D2::Stable, StableStableOp<'b, D1, D2, T1, T2, T>, T1, T2, (T1, T2), T>;

  type Recent = BatchesChain3<
    BatchesJoin<D1::Recent, D2::Stable, RecentStableOp<'b, D1, D2, T1, T2, T>, T1, T2, (T1, T2), T>,
    BatchesJoin<D1::Stable, D2::Recent, StableRecentOp<'b, D1, D2, T1, T2, T>, T1, T2, (T1, T2), T>,
    BatchesJoin<D1::Recent, D2::Recent, RecentRecentOp<'b, D1, D2, T1, T2, T>, T1, T2, (T1, T2), T>,
    (T1, T2),
    T,
  >;

  fn iter_stable(&self) -> Self::Stable {
    let op = ProductOp::new(self.semiring_ctx);
    Self::Stable::join(self.d1.iter_stable(), self.d2.iter_stable(), op)
  }

  fn iter_recent(self) -> Self::Recent {
    let d1_stable = self.d1.iter_stable();
    let d2_stable = self.d2.iter_stable();
    let d1_recent = self.d1.iter_recent();
    let d2_recent = self.d2.iter_recent();
    Self::Recent::chain_3(
      BatchesJoin::join(d1_recent.clone(), d2_stable, ProductOp::new(self.semiring_ctx)),
      BatchesJoin::join(d1_stable, d2_recent.clone(), ProductOp::new(self.semiring_ctx)),
      BatchesJoin::join(d1_recent, d2_recent, ProductOp::new(self.semiring_ctx)),
    )
  }
}

type StableStableOp<'b, D1, D2, T1, T2, T> = ProductOp<
  'b,
  <<D1 as Dataflow<T1, T>>::Stable as Batches<T1, T>>::Batch,
  <<D2 as Dataflow<T2, T>>::Stable as Batches<T2, T>>::Batch,
  T1,
  T2,
  T,
>;

type RecentStableOp<'b, D1, D2, T1, T2, T> = ProductOp<
  'b,
  <<D1 as Dataflow<T1, T>>::Recent as Batches<T1, T>>::Batch,
  <<D2 as Dataflow<T2, T>>::Stable as Batches<T2, T>>::Batch,
  T1,
  T2,
  T,
>;

type StableRecentOp<'b, D1, D2, T1, T2, T> = ProductOp<
  'b,
  <<D1 as Dataflow<T1, T>>::Stable as Batches<T1, T>>::Batch,
  <<D2 as Dataflow<T2, T>>::Recent as Batches<T2, T>>::Batch,
  T1,
  T2,
  T,
>;

type RecentRecentOp<'b, D1, D2, T1, T2, T> = ProductOp<
  'b,
  <<D1 as Dataflow<T1, T>>::Recent as Batches<T1, T>>::Batch,
  <<D2 as Dataflow<T2, T>>::Recent as Batches<T2, T>>::Batch,
  T1,
  T2,
  T,
>;

pub struct ProductOp<'b, I1, I2, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<T1, T>,
  I2: Batch<T2, T>,
{
  semiring_ctx: &'b T::Context,
  phantom: PhantomData<(I1, I2, T1, T2, T)>,
}

impl<'b, I1, I2, T1, T2, T> Clone for ProductOp<'b, I1, I2, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<T1, T>,
  I2: Batch<T2, T>,
{
  fn clone(&self) -> Self {
    Self {
      semiring_ctx: self.semiring_ctx,
      phantom: PhantomData,
    }
  }
}

impl<'b, I1, I2, T1, T2, T> ProductOp<'b, I1, I2, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<T1, T>,
  I2: Batch<T2, T>,
{
  pub fn new(semiring_ctx: &'b T::Context) -> Self {
    Self {
      semiring_ctx,
      phantom: PhantomData,
    }
  }
}

impl<'b, I1, I2, T1, T2, T> BatchBinaryOp<I1, I2> for ProductOp<'b, I1, I2, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<T1, T>,
  I2: Batch<T2, T>,
{
  type IOut = ProductIterator<'b, I1, I2, T1, T2, T>;

  fn apply(&self, mut i1: I1, i2: I2) -> Self::IOut {
    let i1_curr = i1.next();
    Self::IOut {
      i1,
      i1_curr,
      i2_source: i2.clone(),
      i2_clone: i2,
      semiring_ctx: self.semiring_ctx,
      phantom: PhantomData,
    }
  }
}

pub struct ProductIterator<'b, I1, I2, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<T1, T>,
  I2: Batch<T2, T>,
{
  i1: I1,
  i1_curr: Option<StaticElement<T1, T>>,
  i2_source: I2,
  i2_clone: I2,
  semiring_ctx: &'b T::Context,
  phantom: PhantomData<(I1, I2, T1, T2)>,
}

impl<'b, I1, I2, T1, T2, T> Clone for ProductIterator<'b, I1, I2, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<T1, T>,
  I2: Batch<T2, T>,
{
  fn clone(&self) -> Self {
    Self {
      i1: self.i1.clone(),
      i1_curr: self.i1_curr.clone(),
      i2_source: self.i2_source.clone(),
      i2_clone: self.i2_clone.clone(),
      semiring_ctx: self.semiring_ctx,
      phantom: PhantomData,
    }
  }
}

impl<'b, I1, I2, T1, T2, T> Iterator for ProductIterator<'b, I1, I2, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<T1, T>,
  I2: Batch<T2, T>,
{
  type Item = StaticElement<(T1, T2), T>;

  fn next(&mut self) -> Option<Self::Item> {
    loop {
      match &self.i1_curr {
        Some(i1_curr) => match self.i2_clone.next() {
          Some(i2_curr) => {
            let tup = (i1_curr.tuple.get().clone(), i2_curr.tuple.get().clone());
            let tag = self.semiring_ctx.mult(&i1_curr.tag, &i2_curr.tag);
            let elem = StaticElement::new(tup, tag);
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

impl<'b, I1, I2, T1, T2, T> Batch<(T1, T2), T> for ProductIterator<'b, I1, I2, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<T1, T>,
  I2: Batch<T2, T>,
{
}
