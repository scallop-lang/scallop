use std::marker::PhantomData;

use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

use crate::common::foreign_aggregate::Aggregator as DynamicAggregator;
use crate::common::foreign_aggregates::MinMaxAggregator as DynamicMinMaxAggregator;

pub struct ArgminAggregator<T1: StaticTupleTrait, T2: StaticTupleTrait, Prov: Provenance> {
  phantom: PhantomData<(T1, T2, Prov)>,
}

impl<T1: StaticTupleTrait, T2: StaticTupleTrait, Prov: Provenance> ArgminAggregator<T1, T2, Prov> {
  pub fn new() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<T1, T2, T, Prov> Aggregator<T, Prov> for ArgminAggregator<T1, T2, Prov>
where
  T1: StaticTupleTrait + TupleLength,
  T2: StaticTupleTrait,
  T: StaticTupleTrait,
  (T1, T2): FlattenTuple<Output = T>,
  Prov: Provenance,
{
  type Output = T1;

  fn aggregate(
    &self,
    tuples: StaticElements<T, Prov>,
    rt: &RuntimeEnvironment,
    ctx: &Prov,
  ) -> StaticElements<Self::Output, Prov> {
    let agg = DynamicMinMaxAggregator::argmin(<T1 as TupleLength>::len());
    let dyn_elems = tuples
      .into_iter()
      .map(|e| {
        let tag = e.tag.clone();
        DynamicElement::new(T::into_dyn_tuple(e.tuple()), tag)
      })
      .collect();
    let results = agg.aggregate(ctx, rt, dyn_elems);
    let stat_elems = results
      .into_iter()
      .map(|e| StaticElement::new(T1::from_dyn_tuple(e.tuple), e.tag))
      .collect();
    stat_elems
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
