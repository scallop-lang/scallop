use std::marker::PhantomData;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub struct MaxAggregator<Tup: StaticTupleTrait, T: Tag> {
  phantom: PhantomData<(Tup, T)>,
}

impl<Tup: StaticTupleTrait, T: Tag> MaxAggregator<Tup, T> {
  pub fn new() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<Tup, T> Aggregator<Tup, T> for MaxAggregator<Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
{
  type Output = Tup;

  fn aggregate(&self, tuples: StaticElements<Tup, T>, ctx: &T::Context) -> StaticElements<Self::Output, T> {
    ctx.static_max(tuples)
  }
}

impl<Tup, T> Clone for MaxAggregator<Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
{
  fn clone(&self) -> Self {
    Self { phantom: PhantomData }
  }
}
