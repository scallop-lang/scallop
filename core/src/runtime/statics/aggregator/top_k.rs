use std::marker::PhantomData;

use crate::runtime::env::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

use crate::common::foreign_aggregates::*;

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

  fn aggregate(
    &self,
    tuples: StaticElements<Tup, Prov>,
    rt: &RuntimeEnvironment,
    ctx: &Prov,
  ) -> StaticElements<Self::Output, Prov> {
    let agg = TopKSampler::new(self.k);
    let weights = tuples.iter().map(|e| ctx.weight(&e.tag)).collect();
    let indices = agg.sample_weight_only(rt, weights);
    let stat_elems = indices.into_iter().map(|i| tuples[i].clone()).collect();
    stat_elems
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
