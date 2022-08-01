use super::*;

pub struct DynamicAggregationSingleGroupDataflow<'a, T: Tag> {
  pub agg: DynamicAggregator,
  pub d: Box<DynamicDataflow<'a, T>>,
  pub ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for DynamicAggregationSingleGroupDataflow<'a, T> {
  fn clone(&self) -> Self {
    Self {
      agg: self.agg.clone(),
      d: self.d.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, T: Tag> DynamicAggregationSingleGroupDataflow<'a, T> {
  pub fn new(agg: DynamicAggregator, d: DynamicDataflow<'a, T>, ctx: &'a T::Context) -> Self {
    Self {
      agg,
      d: Box::new(d),
      ctx,
    }
  }

  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::empty()
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    if let Some(b) = self.d.iter_recent().next() {
      let batch = b.collect::<Vec<_>>();
      DynamicBatches::single(DynamicBatch::source_vec(self.agg.aggregate(batch, self.ctx)))
    } else {
      DynamicBatches::empty()
    }
  }
}
