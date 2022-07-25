use super::*;
use crate::common::expr::Expr;
use crate::runtime::dynamic::*;

#[derive(Clone)]
pub struct DynamicFilterDataflow<'a, T: Tag> {
  pub source: Box<DynamicDataflow<'a, T>>,
  pub filter: Expr,
}

impl<'a, T: Tag> DynamicFilterDataflow<'a, T> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::filter(self.source.iter_stable(), self.filter.clone())
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::filter(self.source.iter_recent(), self.filter.clone())
  }
}

#[derive(Clone)]
pub struct DynamicFilterBatches<'a, T: Tag> {
  pub source: Box<DynamicBatches<'a, T>>,
  pub filter: Expr,
}

impl<'a, T: Tag> Iterator for DynamicFilterBatches<'a, T> {
  type Item = DynamicBatch<'a, T>;

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
pub struct DynamicFilterBatch<'a, T: Tag> {
  pub source: Box<DynamicBatch<'a, T>>,
  pub filter: Expr,
}

impl<'a, T: Tag> Iterator for DynamicFilterBatch<'a, T> {
  type Item = DynamicElement<T>;

  fn next(&mut self) -> Option<Self::Item> {
    for elem in self.source.by_ref() {
      if self.filter.eval(&elem.tuple).as_bool() {
        return Some(elem);
      }
    }
    None
  }
}
