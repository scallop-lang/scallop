use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;

pub fn union<'b, D1, D2, Tup, Prov>(d1: D1, d2: D2, semiring_ctx: &'b Prov) -> Union<'b, D1, D2, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  D1: Dataflow<Tup, Prov>,
  D2: Dataflow<Tup, Prov>,
{
  Union {
    d1,
    d2,
    semiring_ctx,
    phantom: PhantomData,
  }
}

pub struct Union<'b, D1, D2, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  D1: Dataflow<Tup, Prov>,
  D2: Dataflow<Tup, Prov>,
{
  d1: D1,
  d2: D2,
  semiring_ctx: &'b Prov,
  phantom: PhantomData<Tup>,
}

impl<'b, D1, D2, Tup, Prov> Clone for Union<'b, D1, D2, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  D1: Dataflow<Tup, Prov>,
  D2: Dataflow<Tup, Prov>,
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

impl<'b, D1, D2, Tup, Prov> Dataflow<Tup, Prov> for Union<'b, D1, D2, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  D1: Dataflow<Tup, Prov>,
  D2: Dataflow<Tup, Prov>,
{
  type Stable = BatchesChain<D1::Stable, D2::Stable, Tup, Prov>;

  type Recent = BatchesChain<D1::Recent, D2::Recent, Tup, Prov>;

  fn iter_stable(&self) -> Self::Stable {
    BatchesChain::chain(self.d1.iter_stable(), self.d2.iter_stable())
  }

  fn iter_recent(self) -> Self::Recent {
    BatchesChain::chain(self.d1.iter_recent(), self.d2.iter_recent())
  }
}
