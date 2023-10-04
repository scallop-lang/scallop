use std::marker::PhantomData;

use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

use crate::common::foreign_aggregate::Aggregator as DynamicAggregator;
use crate::common::foreign_aggregates::MinMaxAggregator as DynamicMinMaxAggregator;

pub struct MaxAggregator<Tup: StaticTupleTrait, Prov: Provenance> {
  phantom: PhantomData<(Tup, Prov)>,
}

impl<Tup: StaticTupleTrait, Prov: Provenance> MaxAggregator<Tup, Prov> {
  pub fn new() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<Tup, Prov> Aggregator<Tup, Prov> for MaxAggregator<Tup, Prov>
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
    let agg = DynamicMinMaxAggregator::max();
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
      .map(|e| StaticElement::new(Tup::from_dyn_tuple(e.tuple), e.tag))
      .collect();
    stat_elems
  }
}

impl<Tup, Prov> Clone for MaxAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self { phantom: PhantomData }
  }
}
