use std::marker::PhantomData;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub struct ProdAggregator<Tup: StaticTupleTrait + ProdType, Prov: Provenance> {
  phantom: PhantomData<(Tup, Prov)>,
}

impl<Tup: StaticTupleTrait + ProdType, Prov: Provenance> ProdAggregator<Tup, Prov> {
  pub fn new() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<Tup, Prov> Aggregator<Tup, Prov> for ProdAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait + ProdType,
  Prov: Provenance,
{
  type Output = Tup;

  fn aggregate(&self, tuples: StaticElements<Tup, Prov>, ctx: &Prov) -> StaticElements<Tup, Prov> {
    ctx.static_prod(tuples)
  }
}

impl<Tup, Prov> Clone for ProdAggregator<Tup, Prov>
where
  Tup: StaticTupleTrait + ProdType,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self { phantom: PhantomData }
  }
}
