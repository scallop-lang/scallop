use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct DynamicRecentCollectionDataflow<'a, Prov: Provenance>(pub &'a DynamicCollection<Prov>);

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicRecentCollectionDataflow<'a, Prov> {
  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::single(RefElementsBatch::new(&self.0.elements))
  }
}

#[derive(Clone)]
pub struct DynamicStableCollectionDataflow<'a, Prov: Provenance>(pub &'a DynamicCollection<Prov>);

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicStableCollectionDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::single(RefElementsBatch::new(&self.0.elements))
  }
}
