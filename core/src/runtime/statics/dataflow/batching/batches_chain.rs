use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

#[derive(Clone)]
pub struct BatchesChain<B1, B2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  B1: Batches<Tup, T>,
  B2: Batches<Tup, T>,
{
  b1: B1,
  b2: B2,
  use_b1: bool,
  phantom: PhantomData<(Tup, T)>,
}

impl<B1, B2, Tup, T> BatchesChain<B1, B2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  B1: Batches<Tup, T>,
  B2: Batches<Tup, T>,
{
  pub fn chain(b1: B1, b2: B2) -> Self {
    Self {
      b1: b1,
      b2: b2,
      use_b1: true,
      phantom: PhantomData,
    }
  }
}

impl<B1, B2, Tup, T> Iterator for BatchesChain<B1, B2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  B1: Batches<Tup, T>,
  B2: Batches<Tup, T>,
{
  type Item = EitherBatch<B1::Batch, B2::Batch, Tup, T>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.use_b1 {
      if let Some(b1_curr) = self.b1.next() {
        return Some(EitherBatch::first(b1_curr));
      } else {
        self.use_b1 = false;
      }
    }
    self.b2.next().map(EitherBatch::second)
  }
}

impl<Tup, T, B1, B2> Batches<Tup, T> for BatchesChain<B1, B2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  B1: Batches<Tup, T>,
  B2: Batches<Tup, T>,
{
  type Batch = EitherBatch<B1::Batch, B2::Batch, Tup, T>;
}

pub type BatchesChain3<B1, B2, B3, Tup, T> = BatchesChain<BatchesChain<B1, B2, Tup, T>, B3, Tup, T>;

impl<B1, B2, B3, Tup, T> BatchesChain3<B1, B2, B3, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  B1: Batches<Tup, T>,
  B2: Batches<Tup, T>,
  B3: Batches<Tup, T>,
{
  pub fn chain_3(b1: B1, b2: B2, b3: B3) -> Self {
    BatchesChain::chain(BatchesChain::chain(b1, b2), b3)
  }
}
