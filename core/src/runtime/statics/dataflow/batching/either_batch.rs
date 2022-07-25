use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

#[derive(Clone)]
pub enum EitherBatch<I1, I2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  I1: Batch<Tup, T>,
  I2: Batch<Tup, T>,
{
  First(I1, PhantomData<(Tup, T)>),
  Second(I2, PhantomData<(Tup, T)>),
}

impl<I1, I2, Tup, T> EitherBatch<I1, I2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  I1: Batch<Tup, T>,
  I2: Batch<Tup, T>,
{
  pub fn first(i1: I1) -> Self {
    Self::First(i1, PhantomData)
  }

  pub fn second(i2: I2) -> Self {
    Self::Second(i2, PhantomData)
  }
}

impl<I1, I2, Tup, T> Iterator for EitherBatch<I1, I2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  I1: Batch<Tup, T>,
  I2: Batch<Tup, T>,
{
  type Item = StaticElement<Tup, T>;

  fn next(&mut self) -> Option<Self::Item> {
    match self {
      Self::First(i1, _) => i1.next(),
      Self::Second(i2, _) => i2.next(),
    }
  }
}

impl<I1, I2, Tup, T> Batch<Tup, T> for EitherBatch<I1, I2, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
  I1: Batch<Tup, T>,
  I2: Batch<Tup, T>,
{
}
