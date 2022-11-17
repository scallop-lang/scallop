use std::marker::PhantomData;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub struct ArgmaxAggregator<T1: StaticTupleTrait, T2: StaticTupleTrait, Prov: Provenance> {
  phantom: PhantomData<(T1, T2, Prov)>,
}

impl<T1: StaticTupleTrait, T2: StaticTupleTrait, Prov: Provenance> ArgmaxAggregator<T1, T2, Prov> {
  pub fn new() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<T1: StaticTupleTrait, T2: StaticTupleTrait, Prov> Aggregator<(T1, T2), Prov> for ArgmaxAggregator<T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
{
  type Output = (T1, T2);

  fn aggregate(&self, tuples: StaticElements<(T1, T2), Prov>, ctx: &Prov) -> StaticElements<Self::Output, Prov> {
    ctx.static_argmax(tuples)
  }
}

impl<T1, T2, Prov> Clone for ArgmaxAggregator<T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self { phantom: PhantomData }
  }
}
