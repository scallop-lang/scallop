use std::marker::PhantomData;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub struct ArgminAggregator<T1: StaticTupleTrait, T2: StaticTupleTrait, Prov: Provenance> {
  phantom: PhantomData<(T1, T2, Prov)>,
}

impl<T1: StaticTupleTrait, T2: StaticTupleTrait, Prov: Provenance> ArgminAggregator<T1, T2, Prov> {
  pub fn new() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<T1: StaticTupleTrait, T2: StaticTupleTrait, Prov> Aggregator<(T1, T2), Prov> for ArgminAggregator<T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
{
  type Output = (T1, T2);

  fn aggregate(&self, tuples: StaticElements<(T1, T2), Prov>, ctx: &Prov) -> StaticElements<Self::Output, Prov> {
    ctx.static_argmin(tuples)
  }
}

impl<T1, T2, Prov> Clone for ArgminAggregator<T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self { phantom: PhantomData }
  }
}
