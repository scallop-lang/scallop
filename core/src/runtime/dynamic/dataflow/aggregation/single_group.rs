use super::*;

pub struct DynamicAggregationSingleGroupDataflow<'a, Prov: Provenance> {
  pub agg: DynamicAggregator,
  pub d: Box<DynamicDataflow<'a, Prov>>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicAggregationSingleGroupDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      agg: self.agg.clone(),
      d: self.d.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> DynamicAggregationSingleGroupDataflow<'a, Prov> {
  pub fn new(agg: DynamicAggregator, d: DynamicDataflow<'a, Prov>, ctx: &'a Prov) -> Self {
    Self {
      agg,
      d: Box::new(d),
      ctx,
    }
  }

  pub fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::empty()
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    if let Some(b) = self.d.iter_recent().next() {
      let batch = b.collect::<Vec<_>>();
      DynamicBatches::single(DynamicBatch::source_vec(self.agg.aggregate(batch, self.ctx)))
    } else {
      DynamicBatches::empty()
    }
  }
}
