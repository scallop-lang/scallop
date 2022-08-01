use std::marker::PhantomData;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub struct ArgminAggregator<T1: StaticTupleTrait, T2: StaticTupleTrait, T: Tag> {
  phantom: PhantomData<(T1, T2, T)>,
}

impl<T1: StaticTupleTrait, T2: StaticTupleTrait, T: Tag> ArgminAggregator<T1, T2, T> {
  pub fn new() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<T1: StaticTupleTrait, T2: StaticTupleTrait, T> Aggregator<(T1, T2), T> for ArgminAggregator<T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
{
  type Output = (T1, T2);

  fn aggregate(&self, tuples: StaticElements<(T1, T2), T>, ctx: &T::Context) -> StaticElements<Self::Output, T> {
    ctx.static_argmin(tuples)
  }
}

impl<T1, T2, T> Clone for ArgminAggregator<T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
{
  fn clone(&self) -> Self {
    Self { phantom: PhantomData }
  }
}
