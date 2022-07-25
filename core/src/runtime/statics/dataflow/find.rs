use std::cmp::Ordering;
use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;

pub fn find<D, T1, T2, T>(source: D, key: T1) -> Find<D, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  D: Dataflow<(T1, T2), T>,
{
  Find {
    source,
    key,
    phantom: PhantomData,
  }
}

pub trait FindDataflow<D, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  D: Dataflow<(T1, T2), T>,
{
  fn find(self, key: T1) -> Find<D, T1, T2, T>;
}

impl<D, T1, T2, T> FindDataflow<D, T1, T2, T> for D
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  D: Dataflow<(T1, T2), T>,
{
  fn find(self, key: T1) -> Find<D, T1, T2, T> {
    find(self, key)
  }
}

#[derive(Clone)]
pub struct Find<D, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  D: Dataflow<(T1, T2), T>,
{
  source: D,
  key: T1,
  phantom: PhantomData<(T1, T2, T)>,
}

impl<D, T1, T2, T> Dataflow<(T1, T2), T> for Find<D, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  D: Dataflow<(T1, T2), T>,
{
  type Stable = BatchesMap<D::Stable, FindOp<T1, T2, T>, (T1, T2), (T1, T2), T>;

  type Recent = BatchesMap<D::Recent, FindOp<T1, T2, T>, (T1, T2), (T1, T2), T>;

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
pub struct FindOp<T1, T2, T> {
  key: T1,
  phantom: PhantomData<(T2, T)>,
}

impl<T1, T2, T> FindOp<T1, T2, T> {
  pub fn new(key: T1) -> Self {
    Self {
      key,
      phantom: PhantomData,
    }
  }
}

impl<I, T1, T2, T> BatchUnaryOp<I> for FindOp<T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I: Batch<(T1, T2), T>,
{
  type I2 = FindIterator<I, T1, T2, T>;

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
pub struct FindIterator<I, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I: Batch<(T1, T2), T>,
{
  source_iter: I,
  curr_elem: Option<StaticElement<(T1, T2), T>>,
  key: T1,
  phantom: PhantomData<(T2, T)>,
}

impl<I, T1, T2, T> Iterator for FindIterator<I, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I: Batch<(T1, T2), T>,
{
  type Item = StaticElement<(T1, T2), T>;

  fn next(&mut self) -> Option<Self::Item> {
    let key = self.key.clone();
    loop {
      match &self.curr_elem {
        Some(curr_elem) => match curr_elem.tuple.0 .0.partial_cmp(&self.key).unwrap() {
          Ordering::Less => {
            self.curr_elem = self.source_iter.search_ahead(|x| x.0 .0 < key);
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

impl<I, T1, T2, T> Batch<(T1, T2), T> for FindIterator<I, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I: Batch<(T1, T2), T>,
{
}
