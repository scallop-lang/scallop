use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;

pub fn union<'b, D1, D2, Tup, T>(d1: D1, d2: D2, semiring_ctx: &'b T::Context) -> Union<'b, D1, D2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  D1: Dataflow<Tup, T>,
  D2: Dataflow<Tup, T>,
{
  Union {
    d1,
    d2,
    semiring_ctx,
    phantom: PhantomData,
  }
}

pub struct Union<'b, D1, D2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  D1: Dataflow<Tup, T>,
  D2: Dataflow<Tup, T>,
{
  d1: D1,
  d2: D2,
  semiring_ctx: &'b T::Context,
  phantom: PhantomData<(Tup, T)>,
}

impl<'b, D1, D2, Tup, T> Clone for Union<'b, D1, D2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  D1: Dataflow<Tup, T>,
  D2: Dataflow<Tup, T>,
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

impl<'b, D1, D2, Tup, T> Dataflow<Tup, T> for Union<'b, D1, D2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  D1: Dataflow<Tup, T>,
  D2: Dataflow<Tup, T>,
{
  type Stable = BatchesChain<D1::Stable, D2::Stable, Tup, T>;

  type Recent = BatchesChain<D1::Recent, D2::Recent, Tup, T>;

  fn iter_stable(&self) -> Self::Stable {
    BatchesChain::chain(self.d1.iter_stable(), self.d2.iter_stable())
  }

  fn iter_recent(self) -> Self::Recent {
    BatchesChain::chain(self.d1.iter_recent(), self.d2.iter_recent())
  }
}
