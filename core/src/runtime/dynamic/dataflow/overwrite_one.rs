use super::*;

pub struct DynamicOverwriteOneDataflow<'a, Prov: Provenance> {
  pub source: DynamicDataflow<'a, Prov>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicOverwriteOneDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      source: self.source.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicOverwriteOneDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::new(DynamicOverwriteOneBatches { source: self.source.iter_stable(), ctx: self.ctx })
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::new(DynamicOverwriteOneBatches { source: self.source.iter_recent(), ctx: self.ctx })
  }
}

#[derive(Clone)]
pub struct DynamicOverwriteOneBatches<'a, Prov: Provenance> {
  pub source: DynamicBatches<'a, Prov>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Batches<'a, Prov> for DynamicOverwriteOneBatches<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    self.source.next_batch().map(|batch| {
      DynamicBatch::new(DynamicOverwriteOneBatch {
        source: batch,
        ctx: self.ctx,
      })
    })
  }
}

#[derive(Clone)]
pub struct DynamicOverwriteOneBatch<'a, Prov: Provenance> {
  pub source: DynamicBatch<'a, Prov>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicOverwriteOneBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    self
      .source
      .next_elem()
      .map(|elem| DynamicElement::new(elem.tuple, self.ctx.one()))
  }
}
