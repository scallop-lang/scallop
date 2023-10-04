use std::marker::PhantomData;

use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

use crate::common::foreign_aggregate::Aggregator as DynamicAggregator;
use crate::common::foreign_aggregates::ExistsAggregator as DynamicExistsAggregator;

pub struct ExistsAggregator<Tup: StaticTupleTrait, Prov: Provenance> {
  non_multi_world: bool,
  phantom: PhantomData<(Tup, Prov)>,
}

impl<Tup: StaticTupleTrait, Prov: Provenance> ExistsAggregator<Tup, Prov> {
  pub fn new(non_multi_world: bool) -> Self {
    Self {
      non_multi_world,
      phantom: PhantomData,
    }
  }
}

impl<Tup, Prov> Aggregator<Tup, Prov> for ExistsAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  type Output = bool;

  fn aggregate(
    &self,
    tuples: StaticElements<Tup, Prov>,
    rt: &RuntimeEnvironment,
    ctx: &Prov,
  ) -> StaticElements<Self::Output, Prov> {
    let agg = DynamicExistsAggregator::new(self.non_multi_world);
    let dyn_elems = tuples
      .into_iter()
      .map(|e| {
        let tag = e.tag.clone();
        DynamicElement::new(Tup::into_dyn_tuple(e.tuple()), tag)
      })
      .collect();
    let results = agg.aggregate(ctx, rt, dyn_elems);
    let stat_elems = results
      .into_iter()
      .map(|e| StaticElement::new(e.tuple.as_bool(), e.tag))
      .collect();
    stat_elems
  }
}

impl<Tup, Prov> Clone for ExistsAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self {
      non_multi_world: self.non_multi_world,
      phantom: PhantomData,
    }
  }
}
