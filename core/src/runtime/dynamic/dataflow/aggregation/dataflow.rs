use super::*;

#[derive(Clone)]
pub enum DynamicAggregationDataflow<'a, Prov: Provenance> {
  SingleGroup(DynamicAggregationSingleGroupDataflow<'a, Prov>),
  ImplicitGroup(DynamicAggregationImplicitGroupDataflow<'a, Prov>),
  JoinGroup(DynamicAggregationJoinGroupDataflow<'a, Prov>),
}

impl<'a, Prov: Provenance> Into<DynamicDataflow<'a, Prov>> for DynamicAggregationDataflow<'a, Prov> {
  fn into(self) -> DynamicDataflow<'a, Prov> {
    DynamicDataflow::Aggregate(self)
  }
}

impl<'a, Prov: Provenance> DynamicAggregationDataflow<'a, Prov> {
  pub fn single(agg: DynamicAggregator, d: DynamicDataflow<'a, Prov>, ctx: &'a Prov) -> Self {
    Self::SingleGroup(DynamicAggregationSingleGroupDataflow::new(agg, d, ctx))
  }

  pub fn implicit(agg: DynamicAggregator, d: DynamicDataflow<'a, Prov>, ctx: &'a Prov) -> Self {
    Self::ImplicitGroup(DynamicAggregationImplicitGroupDataflow::new(agg, d, ctx))
  }

  pub fn join(
    agg: DynamicAggregator,
    d1: DynamicDataflow<'a, Prov>,
    d2: DynamicDataflow<'a, Prov>,
    ctx: &'a Prov,
  ) -> Self {
    Self::JoinGroup(DynamicAggregationJoinGroupDataflow::new(agg, d1, d2, ctx))
  }

  pub fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    match self {
      Self::SingleGroup(s) => s.iter_stable(),
      Self::ImplicitGroup(s) => s.iter_stable(),
      Self::JoinGroup(s) => s.iter_stable(),
    }
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    match self {
      Self::SingleGroup(s) => s.iter_recent(),
      Self::ImplicitGroup(s) => s.iter_recent(),
      Self::JoinGroup(s) => s.iter_recent(),
    }
  }
}
