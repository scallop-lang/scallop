use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;

pub fn unit<U: UnitTuple, Prov: Provenance>(ctx: &Prov, first_iteration: bool) -> Unit<U, Prov> {
  Unit {
    ctx,
    first_iteration,
    phantom: PhantomData,
  }
}

pub struct Unit<'a, U: UnitTuple, Prov: Provenance> {
  ctx: &'a Prov,
  first_iteration: bool,
  phantom: PhantomData<U>,
}

impl<'a, U: UnitTuple, Prov: Provenance> Clone for Unit<'a, U, Prov> {
  fn clone(&self) -> Self {
    Self {
      ctx: self.ctx,
      first_iteration: self.first_iteration,
      phantom: PhantomData,
    }
  }
}

impl<'a, U, Prov> Dataflow<U, Prov> for Unit<'a, U, Prov>
where
  U: StaticTupleTrait + UnitTuple,
  Prov: Provenance,
{
  type Stable = SingleBatch<std::iter::Once<StaticElement<U, Prov>>>;

  type Recent = SingleBatch<std::iter::Once<StaticElement<U, Prov>>>;

  fn iter_recent(self) -> Self::Recent {
    if self.first_iteration {
      SingleBatch::singleton(std::iter::once(StaticElement::new(U::unit(), self.ctx.one())))
    } else {
      SingleBatch::empty()
    }
  }

  fn iter_stable(&self) -> Self::Stable {
    if self.first_iteration {
      SingleBatch::empty()
    } else {
      SingleBatch::singleton(std::iter::once(StaticElement::new(U::unit(), self.ctx.one())))
    }
  }
}

pub trait UnitTuple {
  fn unit() -> Self;
}

impl UnitTuple for () {
  fn unit() -> () {
    ()
  }
}

impl<A> UnitTuple for (A,)
where
  A: UnitTuple,
{
  fn unit() -> (A,) {
    (A::unit(),)
  }
}

impl<A, B> UnitTuple for (A, B)
where
  A: UnitTuple,
  B: UnitTuple,
{
  fn unit() -> (A, B) {
    (A::unit(), B::unit())
  }
}

impl<A, B, C> UnitTuple for (A, B, C)
where
  A: UnitTuple,
  B: UnitTuple,
  C: UnitTuple,
{
  fn unit() -> (A, B, C) {
    (A::unit(), B::unit(), C::unit())
  }
}
