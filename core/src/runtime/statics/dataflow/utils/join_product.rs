use crate::runtime::provenance::*;
use crate::runtime::statics::*;

#[derive(Clone)]
pub struct JoinProductIterator<T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
{
  pub v1: Vec<StaticElement<T1, Prov>>,
  pub v2: Vec<StaticElement<T2, Prov>>,
  pub i1: usize,
  pub i2: usize,
}

impl<T1, T2, Prov> JoinProductIterator<T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
{
  pub fn new(v1: Vec<StaticElement<T1, Prov>>, v2: Vec<StaticElement<T2, Prov>>) -> Self {
    Self { v1, v2, i1: 0, i2: 0 }
  }
}

impl<T1, T2, Prov> Iterator for JoinProductIterator<T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
{
  type Item = (StaticElement<T1, Prov>, StaticElement<T2, Prov>);

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
