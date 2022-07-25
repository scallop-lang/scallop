use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct DynamicRecentCollectionDataflow<'a, T: Tag>(pub &'a DynamicCollection<T>);

impl<'a, T: Tag> DynamicRecentCollectionDataflow<'a, T> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::Empty
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::single(DynamicBatch::vec(&self.0.elements))
  }
}

#[derive(Clone)]
pub struct DynamicStableCollectionDataflow<'a, T: Tag>(pub &'a DynamicCollection<T>);

impl<'a, T: Tag> DynamicStableCollectionDataflow<'a, T> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::single(DynamicBatch::vec(&self.0.elements))
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::Empty
  }
}
