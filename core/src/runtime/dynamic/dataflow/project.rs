use super::*;
use crate::common::expr::Expr;

#[derive(Clone)]
pub struct DynamicProjectDataflow<'a, Prov: Provenance> {
  pub source: DynamicDataflow<'a, Prov>,
  pub expression: Expr,
  pub runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicProjectDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::new(DynamicProjectBatches::new(self.runtime, self.source.iter_stable(), self.expression.clone()))
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::new(DynamicProjectBatches::new(self.runtime, self.source.iter_recent(), self.expression.clone()))
  }
}

#[derive(Clone)]
pub struct DynamicProjectBatches<'a, Prov: Provenance> {
  pub runtime: &'a RuntimeEnvironment,
  pub source: DynamicBatches<'a, Prov>,
  pub expression: Expr,
}

impl<'a, Prov: Provenance> DynamicProjectBatches<'a, Prov> {
  pub fn new(runtime: &'a RuntimeEnvironment, source: DynamicBatches<'a, Prov>, expression: Expr) -> Self {
    Self { runtime, source, expression }
  }
}

impl<'a, Prov: Provenance> Batches<'a, Prov> for DynamicProjectBatches<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    match self.source.next_batch() {
      Some(next_batch) => Some(DynamicBatch::new(DynamicProjectBatch {
        runtime: self.runtime,
        source: next_batch,
        expression: self.expression.clone(),
      })),
      None => None,
    }
  }
}

#[derive(Clone)]
pub struct DynamicProjectBatch<'a, Prov: Provenance> {
  pub runtime: &'a RuntimeEnvironment,
  pub source: DynamicBatch<'a, Prov>,
  pub expression: Expr,
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicProjectBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    while let Some(elem) = self.source.next_elem() {
      let val = elem.tuple;
      if let Some(tup) = self.runtime.eval(&self.expression, &val) {
        return Some(DynamicElement::new(tup, elem.tag));
      }
    }
    None
  }
}
