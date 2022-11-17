use super::*;

#[derive(Clone)]
pub struct DynamicUnionDataflow<'a, Prov: Provenance> {
  pub d1: Box<DynamicDataflow<'a, Prov>>,
  pub d2: Box<DynamicDataflow<'a, Prov>>,
}

impl<'a, Prov: Provenance> DynamicUnionDataflow<'a, Prov> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::chain(vec![self.d1.iter_stable(), self.d2.iter_stable()])
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::chain(vec![self.d1.iter_recent(), self.d2.iter_recent()])
  }
}
