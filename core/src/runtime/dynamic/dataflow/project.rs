use super::*;
use crate::common::expr::Expr;
use crate::runtime::dynamic::*;

#[derive(Clone)]
pub struct DynamicProjectDataflow<'a, T: Tag> {
  pub source: Box<DynamicDataflow<'a, T>>,
  pub expression: Expr,
}

impl<'a, T: Tag> DynamicProjectDataflow<'a, T> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::project(self.source.iter_stable(), self.expression.clone())
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::project(self.source.iter_recent(), self.expression.clone())
  }
}

#[derive(Clone)]
pub struct DynamicProjectBatches<'a, T: Tag> {
  pub source: Box<DynamicBatches<'a, T>>,
  pub expression: Expr,
}

impl<'a, T: Tag> Iterator for DynamicProjectBatches<'a, T> {
  type Item = DynamicBatch<'a, T>;

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
pub struct DynamicProjectBatch<'a, T: Tag> {
  pub source: Box<DynamicBatch<'a, T>>,
  pub expression: Expr,
}

impl<'a, T: Tag> Iterator for DynamicProjectBatch<'a, T> {
  type Item = DynamicElement<T>;

  fn next(&mut self) -> Option<Self::Item> {
    self.source.next().map(|elem| {
      let val = elem.tuple;
      DynamicElement::new(self.expression.eval(&val), elem.tag)
    })
  }
}
