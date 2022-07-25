use super::*;
use crate::common::tuple::Tuple;

#[derive(Clone)]
pub struct DynamicFindDataflow<'a, T: Tag> {
  pub source: Box<DynamicDataflow<'a, T>>,
  pub key: Tuple,
}

impl<'a, T: Tag> DynamicFindDataflow<'a, T> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::find(self.source.iter_stable(), self.key.clone())
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::find(self.source.iter_recent(), self.key.clone())
  }
}

#[derive(Clone)]
pub struct DynamicFindBatches<'a, T: Tag> {
  pub source: Box<DynamicBatches<'a, T>>,
  pub key: Tuple,
}

impl<'a, T: Tag> Iterator for DynamicFindBatches<'a, T> {
  type Item = DynamicBatch<'a, T>;

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
pub struct DynamicFindBatch<'a, T: Tag> {
  pub source: Box<DynamicBatch<'a, T>>,
  pub curr_elem: Option<DynamicElement<T>>,
  pub key: Tuple,
}

impl<'a, T: Tag> Iterator for DynamicFindBatch<'a, T> {
  type Item = DynamicElement<T>;

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
