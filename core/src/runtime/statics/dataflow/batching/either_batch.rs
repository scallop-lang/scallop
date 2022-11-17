use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

#[derive(Clone)]
pub enum EitherBatch<I1, I2, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  I1: Batch<Tup, Prov>,
  I2: Batch<Tup, Prov>,
{
  First(I1, PhantomData<(Tup, Prov)>),
  Second(I2, PhantomData<(Tup, Prov)>),
}

impl<I1, I2, Tup, Prov> EitherBatch<I1, I2, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  I1: Batch<Tup, Prov>,
  I2: Batch<Tup, Prov>,
{
  pub fn first(i1: I1) -> Self {
    Self::First(i1, PhantomData)
  }

  pub fn second(i2: I2) -> Self {
    Self::Second(i2, PhantomData)
  }
}

impl<I1, I2, Tup, Prov> Iterator for EitherBatch<I1, I2, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  I1: Batch<Tup, Prov>,
  I2: Batch<Tup, Prov>,
{
  type Item = StaticElement<Tup, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    match self {
      Self::First(i1, _) => i1.next(),
      Self::Second(i2, _) => i2.next(),
    }
  }
}

impl<I1, I2, Tup, Prov> Batch<Tup, Prov> for EitherBatch<I1, I2, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
  I1: Batch<Tup, Prov>,
  I2: Batch<Tup, Prov>,
{
}
