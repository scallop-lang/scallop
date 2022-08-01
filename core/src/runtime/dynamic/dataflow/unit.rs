use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

use super::*;

pub struct DynamicStableUnitDataflow<'a, T: Tag> {
  ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for DynamicStableUnitDataflow<'a, T> {
  fn clone(&self) -> Self {
    Self { ctx: self.ctx }
  }
}

impl<'a, T: Tag> DynamicStableUnitDataflow<'a, T> {
  pub fn new(ctx: &'a T::Context) -> Self {
    Self { ctx }
  }

  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    let elem = DynamicElement::new((), self.ctx.one());
    let batch = DynamicBatch::SourceVec(vec![elem].into_iter());
    DynamicBatches::Single(Some(batch))
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::Empty
  }
}

pub struct DynamicRecentUnitDataflow<'a, T: Tag> {
  ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for DynamicRecentUnitDataflow<'a, T> {
  fn clone(&self) -> Self {
    Self { ctx: self.ctx }
  }
}

impl<'a, T: Tag> DynamicRecentUnitDataflow<'a, T> {
  pub fn new(ctx: &'a T::Context) -> Self {
    Self { ctx }
  }

  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::Empty
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    let elem = DynamicElement::new((), self.ctx.one());
    let batch = DynamicBatch::SourceVec(vec![elem].into_iter());
    DynamicBatches::Single(Some(batch))
  }
}
