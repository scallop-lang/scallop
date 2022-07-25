use super::*;

#[derive(Clone)]
pub struct DynamicUnionDataflow<'a, T: Tag> {
  pub d1: Box<DynamicDataflow<'a, T>>,
  pub d2: Box<DynamicDataflow<'a, T>>,
}

impl<'a, T: Tag> DynamicUnionDataflow<'a, T> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::chain(vec![self.d1.iter_stable(), self.d2.iter_stable()])
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::chain(vec![self.d1.iter_recent(), self.d2.iter_recent()])
  }
}
