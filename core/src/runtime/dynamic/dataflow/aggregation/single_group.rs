use super::*;

pub struct DynamicAggregationSingleGroupDataflow<'a, Prov: Provenance> {
  pub agg: DynamicAggregator<Prov>,
  pub d: DynamicDataflow<'a, Prov>,
  pub ctx: &'a Prov,
  pub runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> Clone for DynamicAggregationSingleGroupDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      agg: self.agg.clone(),
      d: self.d.clone(),
      ctx: self.ctx,
      runtime: self.runtime,
    }
  }
}

impl<'a, Prov: Provenance> DynamicAggregationSingleGroupDataflow<'a, Prov> {
  pub fn new(
    agg: DynamicAggregator<Prov>,
    d: DynamicDataflow<'a, Prov>,
    ctx: &'a Prov,
    runtime: &'a RuntimeEnvironment,
  ) -> Self {
    Self { agg, d, ctx, runtime }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicAggregationSingleGroupDataflow<'a, Prov> {
  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    if let Some(b) = self.d.iter_recent().next_batch() {
      let batch = b.collect::<Vec<_>>();
      DynamicBatches::single(ElementsBatch::new(self.agg.aggregate(self.ctx, self.runtime, batch)))
    } else {
      DynamicBatches::empty()
    }
  }
}
