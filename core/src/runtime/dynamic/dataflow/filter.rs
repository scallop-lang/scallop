use super::*;
use crate::common::expr::Expr;

#[derive(Clone)]
pub struct DynamicFilterDataflow<'a, Prov: Provenance> {
  pub source: DynamicDataflow<'a, Prov>,
  pub filter: Expr,
  pub runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicFilterDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::new(DynamicFilterBatches {
      runtime: self.runtime,
      source: self.source.iter_stable(),
      filter: self.filter.clone(),
    })
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::new(DynamicFilterBatches {
      runtime: self.runtime,
      source: self.source.iter_recent(),
      filter: self.filter.clone(),
    })
  }
}

#[derive(Clone)]
pub struct DynamicFilterBatches<'a, Prov: Provenance> {
  pub runtime: &'a RuntimeEnvironment,
  pub source: DynamicBatches<'a, Prov>,
  pub filter: Expr,
}

impl<'a, Prov: Provenance> Batches<'a, Prov> for DynamicFilterBatches<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    self.source.next_batch().map(|b| {
      DynamicBatch::new(DynamicFilterBatch {
        runtime: self.runtime,
        source: b,
        filter: self.filter.clone(),
      })
    })
  }
}

#[derive(Clone)]
pub struct DynamicFilterBatch<'a, Prov: Provenance> {
  pub runtime: &'a RuntimeEnvironment,
  pub source: DynamicBatch<'a, Prov>,
  pub filter: Expr,
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicFilterBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    while let Some(elem) = self.source.next_elem() {
      if let Some(tup) = self.runtime.eval(&self.filter, &elem.tuple) {
        if tup.as_bool() {
          return Some(elem);
        }
      }
    }
    None
  }
}
