use std::iter::FromIterator;
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
    Self {
      phantom: PhantomData,
    }
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

impl<I, Tup, T> Batches<Tup, T> for EmptyBatches<I>
where
  I: Batch<Tup, T>,
  Tup: StaticTupleTrait,
  T: Tag,
{
  type Batch = I;
}
