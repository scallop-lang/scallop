use std::marker::PhantomData;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub struct MinAggregator<Tup: StaticTupleTrait, Prov: Provenance> {
  phantom: PhantomData<(Tup, Prov)>,
}

impl<Tup: StaticTupleTrait, Prov: Provenance> MinAggregator<Tup, Prov> {
  pub fn new() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<Tup, Prov> Aggregator<Tup, Prov> for MinAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  type Output = Tup;

  fn aggregate(&self, tuples: StaticElements<Tup, Prov>, ctx: &Prov) -> StaticElements<Self::Output, Prov> {
    ctx.static_min(tuples)
  }
}

impl<Tup, Prov> Clone for MinAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self { phantom: PhantomData }
  }
}
