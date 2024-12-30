use super::*;

#[derive(Clone)]
pub struct DynamicDenseMatrixCollection<Prov: Provenance> {
  pub dimension: Vec<usize>,
  pub elements: Vec<Option<Prov::Tag>>,
  pub length: usize,
}

impl<Prov: Provenance> Into<DynamicCollection<Prov>> for DynamicDenseMatrixCollection<Prov> {
  fn into(self) -> DynamicCollection<Prov> {
    DynamicCollection::DenseMatrix(self)
  }
}

impl<Prov: Provenance> DynamicDenseMatrixCollection<Prov> {
  pub fn empty() -> Self {
    Self {
      dimension: vec![],
      elements: vec![],
      length: 0,
    }
  }

  #[allow(unused)]
  pub fn from_vec(elements: Vec<DynamicElement<Prov>>, ctx: &Prov) -> Self {
    if elements.is_empty() {
      Self::empty()
    } else {
      unimplemented!()
    }
  }

  pub fn len(&self) -> usize {
    self.length
  }

  pub fn is_empty(&self) -> bool {
    self.length == 0
  }

  /// Get an element from the indexed vector; this operation is $O(1)$.
  pub fn get(&self, hd_idx: &[usize]) -> Option<Prov::Tag> {
    let elem_idx = self.high_dim_index_to_elem_index(hd_idx);
    self.elements.get(elem_idx).and_then(|e| e.clone())
  }

  #[allow(unused)]
  pub fn high_dim_index_to_elem_index(&self, hd_idx: &[usize]) -> usize {
    unimplemented!()
  }

  pub fn iter<'a>(&'a self) -> DynamicDenseMatrixCollectionIter<'a, Prov> {
    unimplemented!()
  }

  pub fn into_iter(self) -> DynamicDenseMatrixCollectionIntoIter<Prov> {
    unimplemented!()
  }

  pub fn drain<'a>(&'a mut self) -> DynamicDenseMatrixCollectionDrainer<'a, Prov> {
    unimplemented!()
  }
}

impl<Prov: Provenance> std::fmt::Debug for DynamicDenseMatrixCollection<Prov>
where
  Prov::Tag: std::fmt::Debug,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_set().entries(self.iter()).finish()
  }
}

impl<Prov: Provenance> std::fmt::Display for DynamicDenseMatrixCollection<Prov>
where
  DynamicElement<Prov>: std::fmt::Display,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("{")?;
    for (index, elem) in self.iter().enumerate() {
      if index > 0 {
        f.write_str(", ")?;
      }
      std::fmt::Display::fmt(&elem, f)?;
    }
    f.write_str("}")
  }
}

#[allow(unused)]
#[derive(Clone)]
pub struct DynamicDenseMatrixCollectionIter<'a, Prov: Provenance> {
  dimension: Vec<usize>,
  elem_iter: std::iter::Enumerate<core::slice::Iter<'a, Option<Prov::Tag>>>,
}

impl<'a, Prov: Provenance> Iterator for DynamicDenseMatrixCollectionIter<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    unimplemented!()
  }
}

#[allow(unused)]
pub struct DynamicDenseMatrixCollectionIntoIter<Prov: Provenance> {
  dimension: Vec<usize>,
  elem_iter: std::iter::Enumerate<std::vec::IntoIter<Option<Prov::Tag>>>,
}

impl<Prov: Provenance> Iterator for DynamicDenseMatrixCollectionIntoIter<Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    unimplemented!()
  }
}

#[allow(unused)]
pub struct DynamicDenseMatrixCollectionDrainer<'a, Prov: Provenance> {
  dimension: Vec<usize>,
  vec_drainer: std::iter::Enumerate<std::vec::Drain<'a, Option<Prov::Tag>>>,
}

impl<'a, Prov: Provenance> Iterator for DynamicDenseMatrixCollectionDrainer<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    unimplemented!()
  }
}
