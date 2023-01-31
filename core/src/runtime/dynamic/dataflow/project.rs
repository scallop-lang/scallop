use super::*;
use crate::common::expr::Expr;

#[derive(Clone)]
pub struct DynamicProjectDataflow<'a, Prov: Provenance> {
  pub source: Box<DynamicDataflow<'a, Prov>>,
  pub expression: Expr,
}

impl<'a, Prov: Provenance> DynamicProjectDataflow<'a, Prov> {
  pub fn iter_stable(&self, runtime: &'a RuntimeEnvironment) -> DynamicBatches<'a, Prov> {
    DynamicBatches::project(runtime, self.source.iter_stable(runtime), self.expression.clone())
  }

  pub fn iter_recent(&self, runtime: &'a RuntimeEnvironment) -> DynamicBatches<'a, Prov> {
    DynamicBatches::project(runtime, self.source.iter_recent(runtime), self.expression.clone())
  }
}

#[derive(Clone)]
pub struct DynamicProjectBatches<'a, Prov: Provenance> {
  pub runtime: &'a RuntimeEnvironment,
  pub source: Box<DynamicBatches<'a, Prov>>,
  pub expression: Expr,
}

impl<'a, Prov: Provenance> Iterator for DynamicProjectBatches<'a, Prov> {
  type Item = DynamicBatch<'a, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    match self.source.next() {
      Some(next_batch) => Some(DynamicBatch::Project(DynamicProjectBatch {
        runtime: self.runtime,
        source: Box::new(next_batch),
        expression: self.expression.clone(),
      })),
      None => None,
    }
  }
}

#[derive(Clone)]
pub struct DynamicProjectBatch<'a, Prov: Provenance> {
  pub runtime: &'a RuntimeEnvironment,
  pub source: Box<DynamicBatch<'a, Prov>>,
  pub expression: Expr,
}

impl<'a, Prov: Provenance> Iterator for DynamicProjectBatch<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    while let Some(elem) = self.source.next() {
      let val = elem.tuple;
      if let Some(tup) = self.runtime.eval(&self.expression, &val) {
        return Some(DynamicElement::new(tup, elem.tag));
      }
    }
    None
  }
}
