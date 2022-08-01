use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;

pub fn filter<S, F, Tup, T>(source: S, filter_fn: F) -> Filter<S, F, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  S: Dataflow<Tup, T>,
  F: Fn(&Tup) -> bool,
{
  Filter {
    source,
    filter_fn,
    phantom: PhantomData,
  }
}

pub trait FilterOnDataflow<S, F, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  S: Dataflow<Tup, T>,
  F: Fn(&Tup) -> bool,
{
  fn filter(self, filter_fn: F) -> Filter<S, F, Tup, T>;
}

impl<S, F, Tup, T> FilterOnDataflow<S, F, Tup, T> for S
where
  Tup: StaticTupleTrait,
  T: Tag,
  S: Dataflow<Tup, T>,
  F: Fn(&Tup) -> bool,
{
  fn filter(self, filter_fn: F) -> Filter<S, F, Tup, T> {
    filter(self, filter_fn)
  }
}

#[derive(Clone)]
pub struct Filter<S, F, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  S: Dataflow<Tup, T>,
  F: Fn(&Tup) -> bool,
{
  source: S,
  filter_fn: F,
  phantom: PhantomData<(Tup, T)>,
}

impl<S, F, Tup, T> Dataflow<Tup, T> for Filter<S, F, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  S: Dataflow<Tup, T>,
  F: Fn(&Tup) -> bool + Clone,
{
  type Stable = BatchesMap<S::Stable, FilterOp<F, Tup, T>, Tup, Tup, T>;

  type Recent = BatchesMap<S::Recent, FilterOp<F, Tup, T>, Tup, Tup, T>;

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
pub struct FilterOp<F, Tup, T>
where
  F: Fn(&Tup) -> bool + Clone,
{
  filter_fn: F,
  phantom: PhantomData<(Tup, T)>,
}

impl<F, Tup, T> FilterOp<F, Tup, T>
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

impl<I1, F, Tup, T> BatchUnaryOp<I1> for FilterOp<F, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  I1: Batch<Tup, T>,
  F: Fn(&Tup) -> bool + Clone,
{
  type I2 = FilterIterator<I1, F, Tup, T>;

  fn apply(&self, i1: I1) -> Self::I2 {
    Self::I2 {
      source_iter: i1,
      filter_fn: self.filter_fn.clone(),
      phantom: PhantomData,
    }
  }
}

#[derive(Clone)]
pub struct FilterIterator<I, F, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  I: Batch<Tup, T>,
  F: Fn(&Tup) -> bool + Clone,
{
  source_iter: I,
  filter_fn: F,
  phantom: PhantomData<(Tup, T)>,
}

impl<I, F, Tup, T> Iterator for FilterIterator<I, F, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  I: Batch<Tup, T>,
  F: Fn(&Tup) -> bool + Clone,
{
  type Item = StaticElement<Tup, T>;

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

impl<I, F, Tup, T> Batch<Tup, T> for FilterIterator<I, F, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  I: Batch<Tup, T>,
  F: Fn(&Tup) -> bool + Clone,
{
}
