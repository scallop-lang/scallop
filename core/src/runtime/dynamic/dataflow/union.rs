use super::*;

#[derive(Clone)]
pub struct DynamicUnionDataflow<'a, Prov: Provenance> {
  pub d1: DynamicDataflow<'a, Prov>,
  pub d2: DynamicDataflow<'a, Prov>,
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicUnionDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::chain(vec![self.d1.iter_stable(), self.d2.iter_stable()])
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::chain(vec![self.d1.iter_recent(), self.d2.iter_recent()])
  }
}
