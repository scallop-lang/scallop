use super::*;
use crate::common::expr::Expr;
use crate::runtime::dynamic::*;

#[derive(Clone)]
pub struct DynamicFilterDataflow<'a, Prov: Provenance> {
  pub source: Box<DynamicDataflow<'a, Prov>>,
  pub filter: Expr,
}

impl<'a, Prov: Provenance> DynamicFilterDataflow<'a, Prov> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::filter(self.source.iter_stable(), self.filter.clone())
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::filter(self.source.iter_recent(), self.filter.clone())
  }
}

#[derive(Clone)]
pub struct DynamicFilterBatches<'a, Prov: Provenance> {
  pub source: Box<DynamicBatches<'a, Prov>>,
  pub filter: Expr,
}

impl<'a, Prov: Provenance> Iterator for DynamicFilterBatches<'a, Prov> {
  type Item = DynamicBatch<'a, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    self.source.next().map(|b| {
      DynamicBatch::Filter(DynamicFilterBatch {
        source: Box::new(b),
        filter: self.filter.clone(),
      })
    })
  }
}

#[derive(Clone)]
pub struct DynamicFilterBatch<'a, Prov: Provenance> {
  pub source: Box<DynamicBatch<'a, Prov>>,
  pub filter: Expr,
}

impl<'a, Prov: Provenance> Iterator for DynamicFilterBatch<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    for elem in self.source.by_ref() {
      if let Some(tup) = self.filter.eval(&elem.tuple) {
        if tup.as_bool() {
          return Some(elem);
        }
      }
    }
    None
  }
}
