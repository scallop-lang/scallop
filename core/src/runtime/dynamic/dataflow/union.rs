use super::*;
use crate::runtime::env::*;

#[derive(Clone)]
pub struct DynamicUnionDataflow<'a, Prov: Provenance> {
  pub d1: Box<DynamicDataflow<'a, Prov>>,
  pub d2: Box<DynamicDataflow<'a, Prov>>,
}

impl<'a, Prov: Provenance> DynamicUnionDataflow<'a, Prov> {
  pub fn iter_stable(&self, runtime: &'a RuntimeEnvironment) -> DynamicBatches<'a, Prov> {
    DynamicBatches::chain(vec![self.d1.iter_stable(runtime), self.d2.iter_stable(runtime)])
  }

  pub fn iter_recent(&self, runtime: &'a RuntimeEnvironment) -> DynamicBatches<'a, Prov> {
    DynamicBatches::chain(vec![self.d1.iter_recent(runtime), self.d2.iter_recent(runtime)])
  }
}
