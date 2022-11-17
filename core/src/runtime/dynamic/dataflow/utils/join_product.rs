use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

#[derive(Clone)]
pub struct JoinProductIterator<Prov: Provenance> {
  pub v1: Vec<DynamicElement<Prov>>,
  pub v2: Vec<DynamicElement<Prov>>,
  pub i1: usize,
  pub i2: usize,
}

impl<Prov: Provenance> JoinProductIterator<Prov> {
  pub fn new(v1: Vec<DynamicElement<Prov>>, v2: Vec<DynamicElement<Prov>>) -> Self {
    Self { v1, v2, i1: 0, i2: 0 }
  }
}

impl<Prov: Provenance> Iterator for JoinProductIterator<Prov> {
  type Item = (DynamicElement<Prov>, DynamicElement<Prov>);

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
