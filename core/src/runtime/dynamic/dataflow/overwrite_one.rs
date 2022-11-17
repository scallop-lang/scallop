use super::*;

pub struct DynamicOverwriteOneDataflow<'a, Prov: Provenance> {
  pub source: Box<DynamicDataflow<'a, Prov>>,
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

impl<'a, Prov: Provenance> DynamicOverwriteOneDataflow<'a, Prov> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::OverwriteOne(DynamicOverwriteOneBatches {
      source: Box::new(self.source.iter_stable()),
      ctx: self.ctx,
    })
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::OverwriteOne(DynamicOverwriteOneBatches {
      source: Box::new(self.source.iter_recent()),
      ctx: self.ctx,
    })
  }
}

#[derive(Clone)]
pub struct DynamicOverwriteOneBatches<'a, Prov: Provenance> {
  pub source: Box<DynamicBatches<'a, Prov>>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Iterator for DynamicOverwriteOneBatches<'a, Prov> {
  type Item = DynamicBatch<'a, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    self.source.next().map(|batch| {
      DynamicBatch::OverwriteOne(DynamicOverwriteOneBatch {
        source: Box::new(batch),
        ctx: self.ctx,
      })
    })
  }
}

#[derive(Clone)]
pub struct DynamicOverwriteOneBatch<'a, Prov: Provenance> {
  pub source: Box<DynamicBatch<'a, Prov>>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Iterator for DynamicOverwriteOneBatch<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    self.source.next().map(|elem| {
      DynamicElement::new(elem.tuple, self.ctx.one())
    })
  }
}
