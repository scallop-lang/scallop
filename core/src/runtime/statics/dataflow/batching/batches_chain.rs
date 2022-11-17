use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

#[derive(Clone)]
pub struct BatchesChain<B1, B2, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  B1: Batches<Tup, Prov>,
  B2: Batches<Tup, Prov>,
{
  b1: B1,
  b2: B2,
  use_b1: bool,
  phantom: PhantomData<(Tup, Prov)>,
}

impl<B1, B2, Tup, Prov> BatchesChain<B1, B2, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  B1: Batches<Tup, Prov>,
  B2: Batches<Tup, Prov>,
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

impl<B1, B2, Tup, Prov> Iterator for BatchesChain<B1, B2, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  B1: Batches<Tup, Prov>,
  B2: Batches<Tup, Prov>,
{
  type Item = EitherBatch<B1::Batch, B2::Batch, Tup, Prov>;

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

impl<Tup, Prov, B1, B2> Batches<Tup, Prov> for BatchesChain<B1, B2, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  B1: Batches<Tup, Prov>,
  B2: Batches<Tup, Prov>,
{
  type Batch = EitherBatch<B1::Batch, B2::Batch, Tup, Prov>;
}

pub type BatchesChain3<B1, B2, B3, Tup, Prov> = BatchesChain<BatchesChain<B1, B2, Tup, Prov>, B3, Tup, Prov>;

impl<B1, B2, B3, Tup, Prov> BatchesChain3<B1, B2, B3, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  B1: Batches<Tup, Prov>,
  B2: Batches<Tup, Prov>,
  B3: Batches<Tup, Prov>,
{
  pub fn chain_3(b1: B1, b2: B2, b3: B3) -> Self {
    BatchesChain::chain(BatchesChain::chain(b1, b2), b3)
  }
}
