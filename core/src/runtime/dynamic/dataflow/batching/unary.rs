use super::super::*;

pub trait BatchUnaryOp<'a, Prov: Provenance>: dyn_clone::DynClone + 'a {
  fn apply(&self, b: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov>;
}

pub struct DynamicBatchUnaryOp<'a, Prov: Provenance>(Box<dyn BatchUnaryOp<'a, Prov> + 'a>);

impl<'a, Prov: Provenance> DynamicBatchUnaryOp<'a, Prov> {
  pub fn new<Op: BatchUnaryOp<'a, Prov>>(op: Op) -> Self {
    Self(Box::new(op))
  }
}

impl<'a, Prov: Provenance> Clone for DynamicBatchUnaryOp<'a, Prov> {
  fn clone(&self) -> Self {
    Self(dyn_clone::clone_box(&*self.0))
  }
}

impl<'a, Prov: Provenance> BatchUnaryOp<'a, Prov> for DynamicBatchUnaryOp<'a, Prov> {
  fn apply(&self, b: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    self.0.apply(b)
  }
}
