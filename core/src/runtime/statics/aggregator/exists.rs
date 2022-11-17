use std::marker::PhantomData;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub struct ExistsAggregator<Tup: StaticTupleTrait, Prov: Provenance> {
  phantom: PhantomData<(Tup, Prov)>,
}

impl<Tup: StaticTupleTrait, Prov: Provenance> ExistsAggregator<Tup, Prov> {
  pub fn new() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<Tup, Prov> Aggregator<Tup, Prov> for ExistsAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  type Output = bool;

  fn aggregate(&self, tuples: StaticElements<Tup, Prov>, ctx: &Prov) -> StaticElements<Self::Output, Prov> {
    ctx.static_exists(tuples)
  }
}

impl<Tup, Prov> Clone for ExistsAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self { phantom: PhantomData }
  }
}
