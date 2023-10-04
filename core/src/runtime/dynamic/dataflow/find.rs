use super::*;
use crate::common::tuple::Tuple;

#[derive(Clone)]
pub struct DynamicFindDataflow<'a, Prov: Provenance> {
  pub source: DynamicDataflow<'a, Prov>,
  pub key: Tuple,
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicFindDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::new(DynamicFindBatches {
      source: self.source.iter_stable(),
      key: self.key.clone(),
    })
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::new(DynamicFindBatches {
      source: self.source.iter_recent(),
      key: self.key.clone(),
    })
  }
}

#[derive(Clone)]
pub struct DynamicFindBatches<'a, Prov: Provenance> {
  pub source: DynamicBatches<'a, Prov>,
  pub key: Tuple,
}

impl<'a, Prov: Provenance> Batches<'a, Prov> for DynamicFindBatches<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    self.source.next_batch().map(|mut b| {
      let curr_elem = b.next_elem();
      DynamicBatch::new(DynamicFindBatch {
        source: b,
        curr_elem,
        key: self.key.clone(),
      })
    })
  }
}

#[derive(Clone)]
pub struct DynamicFindBatch<'a, Prov: Provenance> {
  pub source: DynamicBatch<'a, Prov>,
  pub curr_elem: Option<DynamicElement<Prov>>,
  pub key: Tuple,
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicFindBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    use std::cmp::Ordering;
    let key = self.key.clone();
    loop {
      match &self.curr_elem {
        Some(elem) => match elem.tuple[0].cmp(&key) {
          Ordering::Less => self.curr_elem = self.source.search_elem_0_until(&key),
          Ordering::Equal => {
            let result = elem.clone();
            self.curr_elem = self.source.next_elem();
            return Some(result);
          }
          Ordering::Greater => return None,
        },
        None => return None,
      }
    }
  }
}
