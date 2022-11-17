use std::marker::PhantomData;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub struct CountAggregator<Tup: StaticTupleTrait, Prov: Provenance> {
  phantom: PhantomData<(Tup, Prov)>,
}

impl<Tup: StaticTupleTrait, Prov: Provenance> CountAggregator<Tup, Prov> {
  pub fn new() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<Tup, Prov> Aggregator<Tup, Prov> for CountAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  type Output = usize;

  fn aggregate(&self, tuples: StaticElements<Tup, Prov>, ctx: &Prov) -> StaticElements<usize, Prov> {
    ctx.static_count(tuples)
  }
}

impl<Tup, Prov> Clone for CountAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self { phantom: PhantomData }
  }
}
