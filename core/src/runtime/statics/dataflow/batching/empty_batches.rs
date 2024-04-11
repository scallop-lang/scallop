use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

#[derive(Clone)]
pub struct EmptyBatches<I>
where
  I: Iterator + Clone,
{
  phantom: PhantomData<I>,
}

impl<I> Default for EmptyBatches<I>
where
  I: Iterator + Clone,
{
  fn default() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<I> Iterator for EmptyBatches<I>
where
  I: Iterator + Clone,
{
  type Item = I;

  fn next(&mut self) -> Option<I> {
    None
  }
}

impl<I> FromIterator<I> for EmptyBatches<I>
where
  I: Iterator + Clone,
{
  fn from_iter<Iter>(_: Iter) -> Self {
    Self::default()
  }
}

impl<I, Tup, Prov> Batches<Tup, Prov> for EmptyBatches<I>
where
  I: Batch<Tup, Prov>,
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  type Batch = I;
}
