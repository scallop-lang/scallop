use std::marker::PhantomData;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub struct SumAggregator<Tup: StaticTupleTrait + SumType, Prov: Provenance> {
  phantom: PhantomData<(Tup, Prov)>,
}

impl<Tup: StaticTupleTrait + SumType, Prov: Provenance> SumAggregator<Tup, Prov> {
  pub fn new() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<Tup, Prov> Aggregator<Tup, Prov> for SumAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait + SumType,
  Prov: Provenance,
{
  type Output = Tup;

  fn aggregate(&self, tuples: StaticElements<Tup, Prov>, ctx: &Prov) -> StaticElements<Tup, Prov> {
    ctx.static_sum(tuples)
  }
}

impl<Tup, Prov> Clone for SumAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait + SumType,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self { phantom: PhantomData }
  }
}
