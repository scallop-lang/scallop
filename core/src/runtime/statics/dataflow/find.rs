use std::cmp::Ordering;
use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;

pub fn find<D, T1, T2, Prov>(source: D, key: T1) -> Find<D, T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
  D: Dataflow<(T1, T2), Prov>,
{
  Find {
    source,
    key,
    phantom: PhantomData,
  }
}

pub trait FindDataflow<D, T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
  D: Dataflow<(T1, T2), Prov>,
{
  fn find(self, key: T1) -> Find<D, T1, T2, Prov>;
}

impl<D, T1, T2, Prov> FindDataflow<D, T1, T2, Prov> for D
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
  D: Dataflow<(T1, T2), Prov>,
{
  fn find(self, key: T1) -> Find<D, T1, T2, Prov> {
    find(self, key)
  }
}

#[derive(Clone)]
pub struct Find<D, T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
  D: Dataflow<(T1, T2), Prov>,
{
  source: D,
  key: T1,
  phantom: PhantomData<(T1, T2, Prov)>,
}

impl<D, T1, T2, Prov> Dataflow<(T1, T2), Prov> for Find<D, T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
  D: Dataflow<(T1, T2), Prov>,
{
  type Stable = BatchesMap<D::Stable, FindOp<T1, T2, Prov>, (T1, T2), (T1, T2), Prov>;

  type Recent = BatchesMap<D::Recent, FindOp<T1, T2, Prov>, (T1, T2), (T1, T2), Prov>;

  fn iter_stable(&self) -> Self::Stable {
    let op = FindOp::new(self.key.clone());
    Self::Stable::new(self.source.iter_stable(), op)
  }

  fn iter_recent(self) -> Self::Recent {
    let op = FindOp::new(self.key.clone());
    Self::Recent::new(self.source.iter_recent(), op)
  }
}

#[derive(Clone)]
pub struct FindOp<T1, T2, Prov> {
  key: T1,
  phantom: PhantomData<(T2, Prov)>,
}

impl<T1, T2, Prov> FindOp<T1, T2, Prov> {
  pub fn new(key: T1) -> Self {
    Self {
      key,
      phantom: PhantomData,
    }
  }
}

impl<I, T1, T2, Prov> BatchUnaryOp<I> for FindOp<T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
  I: Batch<(T1, T2), Prov>,
{
  type I2 = FindIterator<I, T1, T2, Prov>;

  fn apply(&self, mut i1: I) -> Self::I2 {
    let curr_elem = i1.next();
    Self::I2 {
      source_iter: i1,
      curr_elem,
      key: self.key.clone(),
      phantom: PhantomData,
    }
  }
}

#[derive(Clone)]
pub struct FindIterator<I, T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
  I: Batch<(T1, T2), Prov>,
{
  source_iter: I,
  curr_elem: Option<StaticElement<(T1, T2), Prov>>,
  key: T1,
  phantom: PhantomData<(T2, Prov)>,
}

impl<I, T1, T2, Prov> Iterator for FindIterator<I, T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
  I: Batch<(T1, T2), Prov>,
{
  type Item = StaticElement<(T1, T2), Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    let key = self.key.clone();
    loop {
      match &self.curr_elem {
        Some(curr_elem) => match curr_elem.tuple.0.partial_cmp(&self.key).unwrap() {
          Ordering::Less => {
            self.curr_elem = self.source_iter.search_ahead(|x| x.0 < key);
          }
          Ordering::Equal => {
            let result = curr_elem.clone();
            self.curr_elem = self.source_iter.next();
            return Some(result);
          }
          Ordering::Greater => return None,
        },
        None => return None,
      }
    }
  }
}

impl<I, T1, T2, Prov> Batch<(T1, T2), Prov> for FindIterator<I, T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
  I: Batch<(T1, T2), Prov>,
{
}
