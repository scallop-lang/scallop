use std::marker::PhantomData;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub struct CountAggregator<Tup: StaticTupleTrait, T: Tag> {
  phantom: PhantomData<(Tup, T)>,
}

impl<Tup: StaticTupleTrait, T: Tag> CountAggregator<Tup, T> {
  pub fn new() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<Tup, T> Aggregator<Tup, T> for CountAggregator<Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
{
  type Output = usize;

  fn aggregate(&self, tuples: StaticElements<Tup, T>, ctx: &T::Context) -> StaticElements<usize, T> {
    ctx.static_count(tuples)
  }
}

impl<Tup, T> Clone for CountAggregator<Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
{
  fn clone(&self) -> Self {
    Self { phantom: PhantomData }
  }
}
