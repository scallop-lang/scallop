use crate::common::aggregate_op::AggregateOp;
use crate::common::value_type::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DynamicAggregator {
  Count(DynamicCount),
  Sum(DynamicSum),
  Prod(DynamicProd),
  Min(DynamicMin),
  Max(DynamicMax),
  Argmin(DynamicArgmin),
  Argmax(DynamicArgmax),
  Exists(DynamicExists),
  TopK(DynamicTopK),
  CategoricalK(DynamicCategoricalK),
}

impl From<AggregateOp> for DynamicAggregator {
  fn from(o: AggregateOp) -> Self {
    match o {
      AggregateOp::Count => Self::count(),
      AggregateOp::Sum(t) => Self::sum(t),
      AggregateOp::Prod(t) => Self::prod(t),
      AggregateOp::Min => Self::min(),
      AggregateOp::Max => Self::max(),
      AggregateOp::Argmin => Self::argmin(),
      AggregateOp::Argmax => Self::argmax(),
      AggregateOp::Exists => Self::exists(),
      AggregateOp::TopK(k) => Self::top_k(k),
      AggregateOp::CategoricalK(k) => Self::categorical_k(k),
    }
  }
}

impl DynamicAggregator {
  pub fn count() -> Self {
    Self::Count(DynamicCount)
  }

  pub fn sum(ty: ValueType) -> Self {
    Self::Sum(DynamicSum(ty))
  }

  pub fn sum_with_ty<T>() -> Self
  where
    ValueType: FromType<T>,
  {
    Self::Sum(DynamicSum(<ValueType as FromType<T>>::from_type()))
  }

  pub fn prod(ty: ValueType) -> Self {
    Self::Prod(DynamicProd(ty))
  }

  pub fn prod_with_ty<T>() -> Self
  where
    ValueType: FromType<T>,
  {
    Self::Prod(DynamicProd(<ValueType as FromType<T>>::from_type()))
  }

  pub fn min() -> Self {
    Self::Min(DynamicMin)
  }

  pub fn max() -> Self {
    Self::Max(DynamicMax)
  }

  pub fn argmin() -> Self {
    Self::Argmin(DynamicArgmin)
  }

  pub fn argmax() -> Self {
    Self::Argmax(DynamicArgmax)
  }

  pub fn exists() -> Self {
    Self::Exists(DynamicExists)
  }

  pub fn top_k(k: usize) -> Self {
    Self::TopK(DynamicTopK(k))
  }

  pub fn categorical_k(k: usize) -> Self {
    Self::CategoricalK(DynamicCategoricalK(k))
  }

  pub fn aggregate<Prov: Provenance>(
    &self,
    batch: DynamicElements<Prov>,
    ctx: &Prov,
    rt: &RuntimeEnvironment,
  ) -> DynamicElements<Prov> {
    match self {
      Self::Count(c) => c.aggregate(batch, ctx),
      Self::Sum(s) => s.aggregate(batch, ctx),
      Self::Prod(p) => p.aggregate(batch, ctx),
      Self::Min(m) => m.aggregate(batch, ctx),
      Self::Max(m) => m.aggregate(batch, ctx),
      Self::Argmin(m) => m.aggregate(batch, ctx),
      Self::Argmax(m) => m.aggregate(batch, ctx),
      Self::Exists(e) => e.aggregate(batch, ctx),
      Self::TopK(t) => t.aggregate(batch, ctx),
      Self::CategoricalK(c) => c.aggregate(batch, ctx, rt),
    }
  }
}
