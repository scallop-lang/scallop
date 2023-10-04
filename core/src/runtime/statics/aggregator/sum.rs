use std::marker::PhantomData;

use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

use crate::common::foreign_aggregate::Aggregator as DynamicAggregator;
use crate::common::foreign_aggregates::SumProdAggregator as DynamicSumProdAggregator;

pub struct SumAggregator<Tup: StaticTupleTrait + SumType, Prov: Provenance>
where
  ValueType: FromType<Tup>,
{
  phantom: PhantomData<(Tup, Prov)>,
}

impl<Tup: StaticTupleTrait + SumType, Prov: Provenance> SumAggregator<Tup, Prov>
where
  ValueType: FromType<Tup>,
{
  pub fn new() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<Tup, Prov> Aggregator<Tup, Prov> for SumAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait + SumType,
  Prov: Provenance,
  ValueType: FromType<Tup>,
{
  type Output = Tup;

  fn aggregate(
    &self,
    tuples: StaticElements<Tup, Prov>,
    rt: &RuntimeEnvironment,
    ctx: &Prov,
  ) -> StaticElements<Tup, Prov> {
    let agg = DynamicSumProdAggregator::sum::<Tup>(false);
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

impl<Tup, Prov> Clone for SumAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait + SumType,
  Prov: Provenance,
  ValueType: FromType<Tup>,
{
  fn clone(&self) -> Self {
    Self { phantom: PhantomData }
  }
}
