use std::marker::PhantomData;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub struct TopKAggregator<Tup: StaticTupleTrait, Prov: Provenance> {
  k: usize,
  phantom: PhantomData<(Tup, Prov)>,
}

impl<Tup: StaticTupleTrait, Prov: Provenance> TopKAggregator<Tup, Prov> {
  pub fn new(k: usize) -> Self {
    Self {
      k,
      phantom: PhantomData,
    }
  }
}

impl<Tup, Prov> Aggregator<Tup, Prov> for TopKAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  type Output = Tup;

  fn aggregate(&self, tuples: StaticElements<Tup, Prov>, ctx: &Prov) -> StaticElements<Self::Output, Prov> {
    ctx.static_top_k(self.k, tuples)
  }
}

impl<Tup, Prov> Clone for TopKAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self {
      k: self.k,
      phantom: PhantomData,
    }
  }
}
