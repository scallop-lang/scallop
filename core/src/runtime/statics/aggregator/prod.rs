use std::marker::PhantomData;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub struct ProdAggregator<Tup: StaticTupleTrait + ProdType, T: Tag> {
  phantom: PhantomData<(Tup, T)>,
}

impl<Tup: StaticTupleTrait + ProdType, T: Tag> ProdAggregator<Tup, T> {
  pub fn new() -> Self {
    Self { phantom: PhantomData }
  }
}

impl<Tup, T> Aggregator<Tup, T> for ProdAggregator<Tup, T>
where
  Tup: StaticTupleTrait + ProdType,
  T: Tag,
{
  type Output = Tup;

  fn aggregate(&self, tuples: StaticElements<Tup, T>, ctx: &T::Context) -> StaticElements<Tup, T> {
    ctx.static_prod(tuples)
  }
}

impl<Tup, T> Clone for ProdAggregator<Tup, T>
where
  Tup: StaticTupleTrait + ProdType,
  T: Tag,
{
  fn clone(&self) -> Self {
    Self { phantom: PhantomData }
  }
}
