use super::*;
use crate::common::tuple::Tuple;

#[derive(Clone)]
pub struct DynamicFindDataflow<'a, Prov: Provenance> {
  pub source: Box<DynamicDataflow<'a, Prov>>,
  pub key: Tuple,
}

impl<'a, Prov: Provenance> DynamicFindDataflow<'a, Prov> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::find(self.source.iter_stable(), self.key.clone())
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::find(self.source.iter_recent(), self.key.clone())
  }
}

#[derive(Clone)]
pub struct DynamicFindBatches<'a, Prov: Provenance> {
  pub source: Box<DynamicBatches<'a, Prov>>,
  pub key: Tuple,
}

impl<'a, Prov: Provenance> Iterator for DynamicFindBatches<'a, Prov> {
  type Item = DynamicBatch<'a, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    self.source.next().map(|mut b| {
      let curr_elem = b.next();
      DynamicBatch::Find(DynamicFindBatch {
        source: Box::new(b),
        curr_elem,
        key: self.key.clone(),
      })
    })
  }
}

#[derive(Clone)]
pub struct DynamicFindBatch<'a, Prov: Provenance> {
  pub source: Box<DynamicBatch<'a, Prov>>,
  pub curr_elem: Option<DynamicElement<Prov>>,
  pub key: Tuple,
}

impl<'a, Prov: Provenance> Iterator for DynamicFindBatch<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    use std::cmp::Ordering;
    let key = self.key.clone();
    loop {
      match &self.curr_elem {
        Some(elem) => {
          let fst = elem.tuple[0].cmp(&key);
          match fst {
            Ordering::Less => self.curr_elem = self.source.search_ahead(|x| x[0] < key),
            Ordering::Equal => {
              let result = elem.clone();
              self.curr_elem = self.source.next();
              return Some(result);
            }
            Ordering::Greater => return None,
          }
        }
        None => return None,
      }
    }
  }
}
