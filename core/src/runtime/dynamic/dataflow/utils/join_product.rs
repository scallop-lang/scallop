use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

#[derive(Clone)]
pub struct JoinProductIterator<T: Tag> {
  pub v1: Vec<DynamicElement<T>>,
  pub v2: Vec<DynamicElement<T>>,
  pub i1: usize,
  pub i2: usize,
}

impl<T: Tag> JoinProductIterator<T> {
  pub fn new(v1: Vec<DynamicElement<T>>, v2: Vec<DynamicElement<T>>) -> Self {
    Self { v1, v2, i1: 0, i2: 0 }
  }
}

impl<T: Tag> Iterator for JoinProductIterator<T> {
  type Item = (DynamicElement<T>, DynamicElement<T>);

  fn next(&mut self) -> Option<Self::Item> {
    loop {
      if self.i1 < self.v1.len() {
        if self.i2 < self.v2.len() {
          let e1 = &self.v1[self.i1];
          let e2 = &self.v2[self.i2];
          let result = (e1.clone(), e2.clone());
          self.i2 += 1;
          return Some(result);
        } else {
          self.i1 += 1;
          self.i2 = 0;
        }
      } else {
        return None;
      }
    }
  }
}
