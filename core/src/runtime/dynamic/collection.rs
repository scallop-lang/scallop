use super::*;
use crate::runtime::database::*;
use crate::runtime::provenance::*;

#[derive(Clone)]
pub enum DynamicCollection<Prov: Provenance> {
  Sorted(DynamicSortedCollection<Prov>),
  IndexedVec(DynamicIndexedVecCollection<Prov>),
  DenseMatrix(DynamicDenseMatrixCollection<Prov>),
}

impl<Prov: Provenance> std::fmt::Display for DynamicCollection<Prov>
where
  DynamicElement<Prov>: std::fmt::Display
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Sorted(s) => s.fmt(f),
      Self::IndexedVec(s) => s.fmt(f),
      Self::DenseMatrix(s) => s.fmt(f),
    }
  }
}

impl<Prov: Provenance> std::fmt::Debug for DynamicCollection<Prov>
where
  DynamicElement<Prov>: std::fmt::Debug
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Sorted(s) => s.fmt(f),
      Self::IndexedVec(s) => s.fmt(f),
      Self::DenseMatrix(s) => s.fmt(f),
    }
  }
}

impl<Prov: Provenance> DynamicCollection<Prov> {
  pub fn default_empty() -> Self {
    Self::Sorted(DynamicSortedCollection::empty())
  }

  pub fn clone_empty_with_new_prov<Prov2: Provenance>(&self) -> DynamicCollection<Prov2> {
    match self {
      Self::Sorted(_) => DynamicCollection::Sorted(DynamicSortedCollection::empty()),
      Self::IndexedVec(_) => DynamicCollection::IndexedVec(DynamicIndexedVecCollection::empty()),
      Self::DenseMatrix(_) => DynamicCollection::DenseMatrix(DynamicDenseMatrixCollection::empty()),
    }
  }

  pub fn get_metadata(&self) -> StorageMetadata {
    match self {
      Self::Sorted(_) => StorageMetadata::Sorted,
      Self::IndexedVec(_) => StorageMetadata::IndexedVec,
      Self::DenseMatrix(_) => StorageMetadata::DenseMatrix,
    }
  }

  pub fn len(&self) -> usize {
    match self {
      Self::Sorted(s) => s.len(),
      Self::IndexedVec(s) => s.len(),
      Self::DenseMatrix(s) => s.len(),
    }
  }

  pub fn is_empty(&self) -> bool {
    match self {
      Self::Sorted(s) => s.is_empty(),
      Self::IndexedVec(s) => s.is_empty(),
      Self::DenseMatrix(s) => s.is_empty(),
    }
  }

  pub fn as_ref<'a>(&'a self) -> DynamicCollectionRef<'a, Prov> {
    match self {
      Self::Sorted(s) => DynamicCollectionRef::Sorted(&s),
      Self::IndexedVec(s) => DynamicCollectionRef::IndexedVec(&s),
      Self::DenseMatrix(s) => DynamicCollectionRef::DenseMatrix(&s),
    }
  }

  pub fn iter<'a>(&'a self) -> DynamicCollectionIter<'a, Prov> {
    match self {
      Self::Sorted(s) => DynamicCollectionIter::Sorted(s.iter()),
      Self::IndexedVec(s) => DynamicCollectionIter::IndexedVec(s.iter()),
      Self::DenseMatrix(s) => DynamicCollectionIter::DenseMatrix(s.iter()),
    }
  }

  pub fn drain<'a>(&'a mut self) -> DynamicCollectionDrainer<'a, Prov> {
    match self {
      Self::Sorted(s) => DynamicCollectionDrainer::Sorted(s.drain()),
      Self::IndexedVec(s) => DynamicCollectionDrainer::IndexedVec(s.drain()),
      Self::DenseMatrix(s) => DynamicCollectionDrainer::DenseMatrix(s.drain()),
    }
  }
}

#[derive(Clone, Debug)]
pub enum DynamicCollectionRef<'a, Prov: Provenance> {
  Sorted(&'a DynamicSortedCollection<Prov>),
  IndexedVec(&'a DynamicIndexedVecCollection<Prov>),
  DenseMatrix(&'a DynamicDenseMatrixCollection<Prov>),
}

impl<'a, Prov: Provenance> DynamicCollectionRef<'a, Prov> {
  pub fn clone_internal(&self) -> DynamicCollection<Prov> {
    match self {
      Self::Sorted(s) => DynamicCollection::Sorted((*s).clone()),
      Self::IndexedVec(s) => DynamicCollection::IndexedVec((*s).clone()),
      Self::DenseMatrix(s) => DynamicCollection::DenseMatrix((*s).clone()),
    }
  }
}

impl<'a, Prov: Provenance> std::fmt::Display for DynamicCollectionRef<'a, Prov>
where
  DynamicElement<Prov>: std::fmt::Display
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Sorted(s) => s.fmt(f),
      Self::IndexedVec(s) => s.fmt(f),
      Self::DenseMatrix(s) => s.fmt(f),
    }
  }
}

impl<'a, Prov: Provenance> DynamicCollectionRef<'a, Prov> {
  pub fn len(&self) -> usize {
    match self {
      Self::Sorted(s) => s.len(),
      Self::IndexedVec(s) => s.len(),
      Self::DenseMatrix(s) => s.len(),
    }
  }

  pub fn is_empty(&self) -> bool {
    match self {
      Self::Sorted(s) => s.is_empty(),
      Self::IndexedVec(s) => s.is_empty(),
      Self::DenseMatrix(s) => s.is_empty(),
    }
  }

  pub fn iter(&self) -> DynamicCollectionIter<'a, Prov> {
    match self {
      Self::Sorted(s) => DynamicCollectionIter::Sorted(s.iter()),
      Self::IndexedVec(s) => DynamicCollectionIter::IndexedVec(s.iter()),
      Self::DenseMatrix(s) => DynamicCollectionIter::DenseMatrix(s.iter()),
    }
  }
}

#[derive(Clone)]
pub enum DynamicCollectionIter<'a, Prov: Provenance> {
  Sorted(DynamicSortedCollectionIter<'a, Prov>),
  IndexedVec(DynamicIndexedVecCollectionIter<'a, Prov>),
  DenseMatrix(DynamicDenseMatrixCollectionIter<'a, Prov>),
}

impl<'a, Prov: Provenance> Iterator for DynamicCollectionIter<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    match self {
      Self::Sorted(s) => s.next(),
      Self::IndexedVec(s) => s.next(),
      Self::DenseMatrix(s) => s.next(),
    }
  }
}

pub enum DynamicCollectionDrainer<'a, Prov: Provenance> {
  Sorted(DynamicSortedCollectionDrainer<'a, Prov>),
  IndexedVec(DynamicIndexedVecCollectionDrainer<'a, Prov>),
  DenseMatrix(DynamicDenseMatrixCollectionDrainer<'a, Prov>),
}

impl<'a, Prov: Provenance> Iterator for DynamicCollectionDrainer<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    match self {
      Self::Sorted(s) => s.next(),
      Self::IndexedVec(s) => s.next(),
      Self::DenseMatrix(s) => s.next(),
    }
  }
}
