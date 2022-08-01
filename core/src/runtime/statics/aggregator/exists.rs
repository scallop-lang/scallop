use std::marker::PhantomData;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub struct ExistsAggregator<Tup: StaticTupleTrait, T: Tag> {
  phantom: PhantomData<(Tup, T)>,
}

impl<Tup: StaticTupleTrait, T: Tag> ExistsAggregator<Tup, T> {
  pub fn new() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<Tup, T> Aggregator<Tup, T> for ExistsAggregator<Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
{
  type Output = bool;

  fn aggregate(&self, tuples: StaticElements<Tup, T>, ctx: &T::Context) -> StaticElements<Self::Output, T> {
    ctx.static_exists(tuples)
  }
}

impl<Tup, T> Clone for ExistsAggregator<Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
{
  fn clone(&self) -> Self {
    Self { phantom: PhantomData }
  }
}
