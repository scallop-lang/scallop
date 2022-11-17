use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct DynamicRecentCollectionDataflow<'a, Prov: Provenance>(pub &'a DynamicCollection<Prov>);

impl<'a, Prov: Provenance> DynamicRecentCollectionDataflow<'a, Prov> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::Empty
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::single(DynamicBatch::vec(&self.0.elements))
  }
}

#[derive(Clone)]
pub struct DynamicStableCollectionDataflow<'a, Prov: Provenance>(pub &'a DynamicCollection<Prov>);

impl<'a, Prov: Provenance> DynamicStableCollectionDataflow<'a, Prov> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::single(DynamicBatch::vec(&self.0.elements))
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::Empty
  }
}
