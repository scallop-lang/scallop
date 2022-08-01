use super::*;

#[derive(Clone)]
pub enum DynamicAggregationDataflow<'a, T: Tag> {
  SingleGroup(DynamicAggregationSingleGroupDataflow<'a, T>),
  ImplicitGroup(DynamicAggregationImplicitGroupDataflow<'a, T>),
  JoinGroup(DynamicAggregationJoinGroupDataflow<'a, T>),
}

impl<'a, T: Tag> Into<DynamicDataflow<'a, T>> for DynamicAggregationDataflow<'a, T> {
  fn into(self) -> DynamicDataflow<'a, T> {
    DynamicDataflow::Aggregate(self)
  }
}

impl<'a, T: Tag> DynamicAggregationDataflow<'a, T> {
  pub fn single(agg: DynamicAggregator, d: DynamicDataflow<'a, T>, ctx: &'a T::Context) -> Self {
    Self::SingleGroup(DynamicAggregationSingleGroupDataflow::new(agg, d, ctx))
  }

  pub fn implicit(agg: DynamicAggregator, d: DynamicDataflow<'a, T>, ctx: &'a T::Context) -> Self {
    Self::ImplicitGroup(DynamicAggregationImplicitGroupDataflow::new(agg, d, ctx))
  }

  pub fn join(
    agg: DynamicAggregator,
    d1: DynamicDataflow<'a, T>,
    d2: DynamicDataflow<'a, T>,
    ctx: &'a T::Context,
  ) -> Self {
    Self::JoinGroup(DynamicAggregationJoinGroupDataflow::new(agg, d1, d2, ctx))
  }

  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    match self {
      Self::SingleGroup(s) => s.iter_stable(),
      Self::ImplicitGroup(s) => s.iter_stable(),
      Self::JoinGroup(s) => s.iter_stable(),
    }
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    match self {
      Self::SingleGroup(s) => s.iter_recent(),
      Self::ImplicitGroup(s) => s.iter_recent(),
      Self::JoinGroup(s) => s.iter_recent(),
    }
  }
}
