use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;

pub fn sorted<S, Tup, Prov>(source: S) -> Sorted<S, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  S: Dataflow<Tup, Prov>,
{
  Sorted {
    source,
    phantom: PhantomData,
  }
}

#[derive(Clone)]
pub struct Sorted<S, T, Prov>
where
  T: StaticTupleTrait,
  Prov: Provenance,
  S: Dataflow<T, Prov>,
{
  source: S,
  phantom: PhantomData<(T, Prov)>
}

impl<S, T, Prov> Dataflow<T, Prov> for Sorted<S, T, Prov>
where
  T: StaticTupleTrait,
  Prov: Provenance,
  S: Dataflow<T, Prov>,
{
  type Stable = BatchesMap<S::Stable, SortedOp<T, Prov>, T, T, Prov>;

  type Recent = BatchesMap<S::Recent, SortedOp<T, Prov>, T, T, Prov>;

  fn iter_stable(&self) -> Self::Stable {
    let op = SortedOp::new();
    Self::Stable::new(self.source.iter_stable(), op)
  }

  fn iter_recent(self) -> Self::Recent {
    let op = SortedOp::new();
    Self::Recent::new(self.source.iter_recent(), op)
  }
}

#[derive(Clone)]
pub struct SortedOp<T, Prov>(PhantomData<(T, Prov)>);

impl<T: StaticTupleTrait, Prov: Provenance> SortedOp<T, Prov> {
  pub fn new() -> Self {
    Self(PhantomData)
  }
}

impl<I, T, Prov> BatchUnaryOp<I> for SortedOp<T, Prov>
where
  T: StaticTupleTrait,
  Prov: Provenance,
  I: Batch<T, Prov>,
{
  type I2 = SortedIterator<I, T, Prov>;

  fn apply(&self, i1: I) -> Self::I2 {
    SortedIterator {
      source: i1,
      sorted_tgt: None,
    }
  }
}

#[derive(Clone)]
pub struct SortedIterator<I, T, Prov>
where
  T: StaticTupleTrait,
  Prov: Provenance,
  I: Batch<T, Prov>,
{
  source: I,
  sorted_tgt: Option<std::vec::IntoIter<StaticElement<T, Prov>>>,
}

impl<I, T, Prov> Iterator for SortedIterator<I, T, Prov>
where
  T: StaticTupleTrait,
  Prov: Provenance,
  I: Batch<T, Prov>,
{
  type Item = StaticElement<T, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    if let Some(iterator) = &mut self.sorted_tgt {
      iterator.next()
    } else {
      let mut tgt = self.source.collect_vec();
      tgt.sort();
      let mut sorted_tgt_iter = tgt.into_iter();
      let maybe_first_elem = sorted_tgt_iter.next();
      if let Some(first_elem) = maybe_first_elem {
        self.sorted_tgt = Some(sorted_tgt_iter);
        Some(first_elem)
      } else {
        None
      }
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    if let Some(sorted_tgt) = &self.sorted_tgt {
      sorted_tgt.size_hint()
    } else {
      (0, None)
    }
  }
}

impl<I, T, Prov> Batch<T, Prov> for SortedIterator<I, T, Prov>
where
  I: Batch<T, Prov>,
  T: StaticTupleTrait,
  Prov: Provenance,
{
}
