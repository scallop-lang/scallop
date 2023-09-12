use super::*;
use crate::common::tuple_type::*;

pub struct DynamicStableUnitDataflow<'a, Prov: Provenance> {
  ctx: &'a Prov,
  tuple_type: TupleType,
}

impl<'a, Prov: Provenance> Clone for DynamicStableUnitDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      ctx: self.ctx,
      tuple_type: self.tuple_type.clone(),
    }
  }
}

impl<'a, Prov: Provenance> DynamicStableUnitDataflow<'a, Prov> {
  pub fn new(ctx: &'a Prov, tuple_type: TupleType) -> Self {
    Self { ctx, tuple_type }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicStableUnitDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    let elem = DynamicElement::new(self.tuple_type.unit_value(), self.ctx.one());
    let batch = ElementsBatch::new(vec![elem]);
    DynamicBatches::single(batch)
  }
}

pub struct DynamicRecentUnitDataflow<'a, Prov: Provenance> {
  ctx: &'a Prov,
  tuple_type: TupleType,
}

impl<'a, Prov: Provenance> Clone for DynamicRecentUnitDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      ctx: self.ctx,
      tuple_type: self.tuple_type.clone(),
    }
  }
}

impl<'a, Prov: Provenance> DynamicRecentUnitDataflow<'a, Prov> {
  pub fn new(ctx: &'a Prov, tuple_type: TupleType) -> Self {
    Self { ctx, tuple_type }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicRecentUnitDataflow<'a, Prov> {
  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    let elem = DynamicElement::new(self.tuple_type.unit_value(), self.ctx.one());
    let batch = ElementsBatch::new(vec![elem]);
    DynamicBatches::single(batch)
  }
}
