use std::marker::PhantomData;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub struct SumAggregator<Tup: StaticTupleTrait + SumType, T: Tag> {
  phantom: PhantomData<(Tup, T)>,
}

impl<Tup: StaticTupleTrait + SumType, T: Tag> SumAggregator<Tup, T> {
  pub fn new() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<Tup, T> Aggregator<Tup, T> for SumAggregator<Tup, T>
where
  Tup: StaticTupleTrait + SumType,
  T: Tag,
{
  type Output = Tup;

  fn aggregate(&self, tuples: StaticElements<Tup, T>, ctx: &T::Context) -> StaticElements<Tup, T> {
    ctx.static_sum(tuples)
  }
}

impl<Tup, T> Clone for SumAggregator<Tup, T>
where
  Tup: StaticTupleTrait + SumType,
  T: Tag,
{
  fn clone(&self) -> Self {
    Self { phantom: PhantomData }
  }
}
