use super::*;
use crate::common::expr::Expr;
use crate::runtime::dynamic::*;

#[derive(Clone)]
pub struct DynamicProjectDataflow<'a, Prov: Provenance> {
  pub source: Box<DynamicDataflow<'a, Prov>>,
  pub expression: Expr,
}

impl<'a, Prov: Provenance> DynamicProjectDataflow<'a, Prov> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::project(self.source.iter_stable(), self.expression.clone())
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::project(self.source.iter_recent(), self.expression.clone())
  }
}

#[derive(Clone)]
pub struct DynamicProjectBatches<'a, Prov: Provenance> {
  pub source: Box<DynamicBatches<'a, Prov>>,
  pub expression: Expr,
}

impl<'a, Prov: Provenance> Iterator for DynamicProjectBatches<'a, Prov> {
  type Item = DynamicBatch<'a, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    match self.source.next() {
      Some(next_batch) => Some(DynamicBatch::Project(DynamicProjectBatch {
        source: Box::new(next_batch),
        expression: self.expression.clone(),
      })),
      None => None,
    }
  }
}

#[derive(Clone)]
pub struct DynamicProjectBatch<'a, Prov: Provenance> {
  pub source: Box<DynamicBatch<'a, Prov>>,
  pub expression: Expr,
}

impl<'a, Prov: Provenance> Iterator for DynamicProjectBatch<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    while let Some(elem) = self.source.next() {
      let val = elem.tuple;
      if let Some(tup) = self.expression.eval(&val) {
        return Some(DynamicElement::new(tup, elem.tag));
      }
    }
    None
  }
}
