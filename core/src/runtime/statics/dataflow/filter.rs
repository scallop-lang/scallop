use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;

pub fn filter<S, F, Tup, Prov>(source: S, filter_fn: F) -> Filter<S, F, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  S: Dataflow<Tup, Prov>,
  F: Fn(&Tup) -> bool,
{
  Filter {
    source,
    filter_fn,
    phantom: PhantomData,
  }
}

pub trait FilterOnDataflow<S, F, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  S: Dataflow<Tup, Prov>,
  F: Fn(&Tup) -> bool,
{
  fn filter(self, filter_fn: F) -> Filter<S, F, Tup, Prov>;
}

impl<S, F, Tup, Prov> FilterOnDataflow<S, F, Tup, Prov> for S
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  S: Dataflow<Tup, Prov>,
  F: Fn(&Tup) -> bool,
{
  fn filter(self, filter_fn: F) -> Filter<S, F, Tup, Prov> {
    filter(self, filter_fn)
  }
}

#[derive(Clone)]
pub struct Filter<S, F, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  S: Dataflow<Tup, Prov>,
  F: Fn(&Tup) -> bool,
{
  source: S,
  filter_fn: F,
  phantom: PhantomData<(Tup, Prov)>,
}

impl<S, F, Tup, Prov> Dataflow<Tup, Prov> for Filter<S, F, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  S: Dataflow<Tup, Prov>,
  F: Fn(&Tup) -> bool + Clone,
{
  type Stable = BatchesMap<S::Stable, FilterOp<F, Tup, Prov>, Tup, Tup, Prov>;

  type Recent = BatchesMap<S::Recent, FilterOp<F, Tup, Prov>, Tup, Tup, Prov>;

  fn iter_stable(&self) -> Self::Stable {
    let op = FilterOp::new(self.filter_fn.clone());
    Self::Stable::new(self.source.iter_stable(), op)
  }

  fn iter_recent(self) -> Self::Recent {
    let op = FilterOp::new(self.filter_fn);
    Self::Recent::new(self.source.iter_recent(), op)
  }
}

#[derive(Clone)]
pub struct FilterOp<F, Tup, Prov>
where
  F: Fn(&Tup) -> bool + Clone,
{
  filter_fn: F,
  phantom: PhantomData<(Tup, Prov)>,
}

impl<F, Tup, Prov> FilterOp<F, Tup, Prov>
where
  F: Fn(&Tup) -> bool + Clone,
{
  pub fn new(filter_fn: F) -> Self {
    Self {
      filter_fn,
      phantom: PhantomData,
    }
  }
}

impl<I1, F, Tup, Prov> BatchUnaryOp<I1> for FilterOp<F, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  I1: Batch<Tup, Prov>,
  F: Fn(&Tup) -> bool + Clone,
{
  type I2 = FilterIterator<I1, F, Tup, Prov>;

  fn apply(&self, i1: I1) -> Self::I2 {
    Self::I2 {
      source_iter: i1,
      filter_fn: self.filter_fn.clone(),
      phantom: PhantomData,
    }
  }
}

#[derive(Clone)]
pub struct FilterIterator<I, F, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  I: Batch<Tup, Prov>,
  F: Fn(&Tup) -> bool + Clone,
{
  source_iter: I,
  filter_fn: F,
  phantom: PhantomData<(Tup, Prov)>,
}

impl<I, F, Tup, Prov> Iterator for FilterIterator<I, F, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  I: Batch<Tup, Prov>,
  F: Fn(&Tup) -> bool + Clone,
{
  type Item = StaticElement<Tup, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    loop {
      match self.source_iter.next() {
        Some(item) => {
          if (self.filter_fn)(item.tuple.get()) {
            return Some(item);
          }
        }
        None => return None,
      }
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    self.source_iter.size_hint()
  }
}

impl<I, F, Tup, Prov> Batch<Tup, Prov> for FilterIterator<I, F, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  I: Batch<Tup, Prov>,
  F: Fn(&Tup) -> bool + Clone,
{
}
