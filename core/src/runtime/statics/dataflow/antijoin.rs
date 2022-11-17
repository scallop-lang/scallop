use std::cmp::Ordering;
use std::marker::PhantomData;

use super::utils::*;
use super::*;
use crate::runtime::provenance::*;

pub fn antijoin<'b, D1, D2, K, T1, Prov>(d1: D1, d2: D2, semiring_ctx: &'b Prov) -> Antijoin<'b, D1, D2, K, T1, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  Prov: Provenance,
  D1: Dataflow<(K, T1), Prov>,
  D2: Dataflow<K, Prov>,
{
  Antijoin {
    d1,
    d2,
    semiring_ctx,
    phantom: PhantomData,
  }
}

pub struct Antijoin<'b, D1, D2, K, T1, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  D1: Dataflow<(K, T1), Prov>,
  D2: Dataflow<K, Prov>,
  Prov: Provenance,
{
  d1: D1,
  d2: D2,
  semiring_ctx: &'b Prov,
  phantom: PhantomData<(K, T1, Prov)>,
}

impl<'b, D1, D2, K, T1, Prov> Clone for Antijoin<'b, D1, D2, K, T1, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  D1: Dataflow<(K, T1), Prov>,
  D2: Dataflow<K, Prov>,
  Prov: Provenance,
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

impl<'b, D1, D2, K, T1, Prov> Dataflow<(K, T1), Prov> for Antijoin<'b, D1, D2, K, T1, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  D1: Dataflow<(K, T1), Prov>,
  D2: Dataflow<K, Prov>,
  Prov: Provenance,
{
  type Stable = BatchesJoin<D1::Stable, D2::Stable, StableStableOp<'b, D1, D2, K, T1, Prov>, (K, T1), K, (K, T1), Prov>;

  type Recent = BatchesChain3<
    BatchesJoin<D1::Recent, D2::Stable, RecentStableOp<'b, D1, D2, K, T1, Prov>, (K, T1), K, (K, T1), Prov>,
    BatchesJoin<D1::Stable, D2::Recent, StableRecentOp<'b, D1, D2, K, T1, Prov>, (K, T1), K, (K, T1), Prov>,
    BatchesJoin<D1::Recent, D2::Recent, RecentRecentOp<'b, D1, D2, K, T1, Prov>, (K, T1), K, (K, T1), Prov>,
    (K, T1),
    Prov,
  >;

  fn iter_stable(&self) -> Self::Stable {
    let op = AntijoinOp::new(self.semiring_ctx);
    Self::Stable::join(self.d1.iter_stable(), self.d2.iter_stable(), op)
  }

  fn iter_recent(self) -> Self::Recent {
    let d1_stable = self.d1.iter_stable();
    let d2_stable = self.d2.iter_stable();
    let d1_recent = self.d1.iter_recent();
    let d2_recent = self.d2.iter_recent();
    Self::Recent::chain_3(
      BatchesJoin::join(d1_recent.clone(), d2_stable, AntijoinOp::new(self.semiring_ctx)),
      BatchesJoin::join(d1_stable, d2_recent.clone(), AntijoinOp::new(self.semiring_ctx)),
      BatchesJoin::join(d1_recent, d2_recent, AntijoinOp::new(self.semiring_ctx)),
    )
  }
}

type StableStableOp<'b, D1, D2, K, T1, Prov> = AntijoinOp<
  'b,
  <<D1 as Dataflow<(K, T1), Prov>>::Stable as Batches<(K, T1), Prov>>::Batch,
  <<D2 as Dataflow<K, Prov>>::Stable as Batches<K, Prov>>::Batch,
  K,
  T1,
  Prov,
>;

type RecentStableOp<'b, D1, D2, K, T1, Prov> = AntijoinOp<
  'b,
  <<D1 as Dataflow<(K, T1), Prov>>::Recent as Batches<(K, T1), Prov>>::Batch,
  <<D2 as Dataflow<K, Prov>>::Stable as Batches<K, Prov>>::Batch,
  K,
  T1,
  Prov,
>;

type StableRecentOp<'b, D1, D2, K, T1, Prov> = AntijoinOp<
  'b,
  <<D1 as Dataflow<(K, T1), Prov>>::Stable as Batches<(K, T1), Prov>>::Batch,
  <<D2 as Dataflow<K, Prov>>::Recent as Batches<K, Prov>>::Batch,
  K,
  T1,
  Prov,
>;

type RecentRecentOp<'b, D1, D2, K, T1, Prov> = AntijoinOp<
  'b,
  <<D1 as Dataflow<(K, T1), Prov>>::Recent as Batches<(K, T1), Prov>>::Batch,
  <<D2 as Dataflow<K, Prov>>::Recent as Batches<K, Prov>>::Batch,
  K,
  T1,
  Prov,
>;

pub struct AntijoinOp<'a, I1, I2, K, T1, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  I1: Batch<(K, T1), Prov>,
  I2: Batch<K, Prov>,
  Prov: Provenance,
{
  semiring_ctx: &'a Prov,
  phantom: PhantomData<(I1, I2, K, T1, Prov)>,
}

impl<'a, I1, I2, K, T1, Prov> Clone for AntijoinOp<'a, I1, I2, K, T1, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  I1: Batch<(K, T1), Prov>,
  I2: Batch<K, Prov>,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self {
      semiring_ctx: self.semiring_ctx,
      phantom: PhantomData,
    }
  }
}

impl<'a, I1, I2, K, T1, Prov> AntijoinOp<'a, I1, I2, K, T1, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  I1: Batch<(K, T1), Prov>,
  I2: Batch<K, Prov>,
  Prov: Provenance,
{
  pub fn new(semiring_ctx: &'a Prov) -> Self {
    Self {
      semiring_ctx,
      phantom: PhantomData,
    }
  }
}

impl<'a, I1, I2, K, T1, Prov> BatchBinaryOp<I1, I2> for AntijoinOp<'a, I1, I2, K, T1, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  Prov: Provenance,
  I1: Batch<(K, T1), Prov>,
  I2: Batch<K, Prov>,
{
  type IOut = AntijoinIterator<'a, I1, I2, K, T1, Prov>;

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

pub struct AntijoinIterator<'b, I1, I2, K, T1, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  Prov: Provenance,
  I1: Batch<(K, T1), Prov>,
  I2: Batch<K, Prov>,
{
  i1: I1,
  i2: I2,
  i1_curr: Option<StaticElement<(K, T1), Prov>>,
  i2_curr: Option<StaticElement<K, Prov>>,
  curr_iter: Option<JoinProductIterator<(K, T1), K, Prov>>,
  semiring_ctx: &'b Prov,
}

impl<'b, I1, I2, K, T1, Prov> Clone for AntijoinIterator<'b, I1, I2, K, T1, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  Prov: Provenance,
  I1: Batch<(K, T1), Prov>,
  I2: Batch<K, Prov>,
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

impl<'b, I1, I2, K, T1, Prov> Iterator for AntijoinIterator<'b, I1, I2, K, T1, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  Prov: Provenance,
  I1: Batch<(K, T1), Prov>,
  I2: Batch<K, Prov>,
{
  type Item = StaticElement<(K, T1), Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    loop {
      // First go through curr joint product iterator
      if let Some(curr_iter) = &mut self.curr_iter {
        if let Some((e1, e2)) = curr_iter.next() {
          let maybe_tag = self.semiring_ctx.minus(&e1.tag, &e2.tag);
          if let Some(tag) = maybe_tag {
            let result = StaticElement::new(e1.tuple().clone(), tag);
            return Some(result);
          } else {
            continue;
          }
        } else {
          self.i1.step(curr_iter.v1.len() - 1);
          self.i1_curr = self.i1.next();
          self.i2.step(curr_iter.v2.len() - 1);
          self.i2_curr = self.i2.next();
          self.curr_iter = None;
        }
      }

      // Then continue
      match (&self.i1_curr, &self.i2_curr) {
        (Some(i1_curr), Some(i2_curr)) => match i1_curr.tuple.0.partial_cmp(&i2_curr.tuple).unwrap() {
          Ordering::Less => {
            let result = i1_curr.clone();
            self.i1_curr = self.i1.next();
            return Some(result);
          }
          Ordering::Equal => {
            let key = &i1_curr.tuple.0;
            let v1 = std::iter::once(i1_curr.clone())
              .chain(self.i1.clone().take_while(|x| &x.tuple.0 == key))
              .collect::<Vec<_>>();
            let v2 = std::iter::once(i2_curr.clone()).collect::<Vec<_>>();
            let iter = JoinProductIterator::new(v1, v2);
            self.curr_iter = Some(iter);
          }
          Ordering::Greater => self.i2_curr = self.i2.search_ahead(|i2_next| i2_next < &i1_curr.tuple.0),
        },
        (Some(i1_curr), None) => {
          let result = i1_curr.clone();
          self.i1_curr = self.i1.next();
          return Some(result);
        }
        _ => break None,
      }
    }
  }
}

impl<'b, I1, I2, K, T1, Prov> Batch<(K, T1), Prov> for AntijoinIterator<'b, I1, I2, K, T1, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  Prov: Provenance,
  I1: Batch<(K, T1), Prov>,
  I2: Batch<K, Prov>,
{
}
