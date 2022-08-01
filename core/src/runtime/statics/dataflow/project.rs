use std::marker::PhantomData;

use super::*;
use crate::runtime::statics::*;

pub fn project<S, F, T1, T2, T>(source: S, map_fn: F) -> Projection<S, F, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  S: Dataflow<T1, T>,
  F: Fn(T1) -> T2,
{
  Projection {
    source,
    map_fn,
    phantom: PhantomData,
  }
}

pub trait ProjectionOnDataflow<S, F, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  S: Dataflow<T1, T>,
  F: Fn(T1) -> T2,
{
  fn project(self, map_fn: F) -> Projection<S, F, T1, T2, T>;
}

impl<S, F, T1, T2, T> ProjectionOnDataflow<S, F, T1, T2, T> for S
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  S: Dataflow<T1, T>,
  F: Fn(T1) -> T2,
{
  fn project(self, map_fn: F) -> Projection<S, F, T1, T2, T> {
    project(self, map_fn)
  }
}

#[derive(Clone)]
pub struct Projection<S, F, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  S: Dataflow<T1, T>,
  F: Fn(T1) -> T2,
{
  source: S,
  map_fn: F,
  phantom: PhantomData<(T1, T2, T)>,
}

impl<S, F, T1, T2, T> Dataflow<T2, T> for Projection<S, F, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  S: Dataflow<T1, T>,
  F: Fn(T1) -> T2 + Clone,
{
  type Stable = BatchesMap<S::Stable, ProjectOp<F, T1, T2, T>, T1, T2, T>;

  type Recent = BatchesMap<S::Recent, ProjectOp<F, T1, T2, T>, T1, T2, T>;

  fn iter_stable(&self) -> Self::Stable {
    let op = ProjectOp::new(self.map_fn.clone());
    Self::Stable::new(self.source.iter_stable(), op)
  }

  fn iter_recent(self) -> Self::Recent {
    let op = ProjectOp::new(self.map_fn.clone());
    Self::Recent::new(self.source.iter_recent(), op)
  }
}

#[derive(Clone)]
pub struct ProjectOp<F, T1, T2, Tag>
where
  F: Fn(T1) -> T2 + Clone,
{
  map_fn: F,
  phantom: PhantomData<(T1, T2, Tag)>,
}

impl<F, T1, T2, Tag> ProjectOp<F, T1, T2, Tag>
where
  F: Fn(T1) -> T2 + Clone,
{
  pub fn new(map_fn: F) -> Self {
    Self {
      map_fn,
      phantom: PhantomData,
    }
  }
}

impl<I1, F, T1, T2, T> BatchUnaryOp<I1> for ProjectOp<F, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I1: Batch<T1, T>,
  F: Fn(T1) -> T2 + Clone,
{
  type I2 = ProjectionIterator<I1, F, T1, T2, T>;

  fn apply(&self, i1: I1) -> Self::I2 {
    Self::I2 {
      source_iter: i1,
      map_fn: self.map_fn.clone(),
      phantom: PhantomData,
    }
  }
}

#[derive(Clone)]
pub struct ProjectionIterator<I, F, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I: Batch<T1, T>,
  F: Fn(T1) -> T2 + Clone,
{
  source_iter: I,
  map_fn: F,
  phantom: PhantomData<(T1, T2, T)>,
}

impl<I, F, T1, T2, T> Iterator for ProjectionIterator<I, F, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I: Batch<T1, T>,
  F: Fn(T1) -> T2 + Clone,
{
  type Item = StaticElement<T2, T>;

  fn next(&mut self) -> Option<Self::Item> {
    match self.source_iter.next() {
      Some(item) => {
        let (tuple, tag) = item.into();
        Some(StaticElement::new((self.map_fn)(tuple), tag))
      }
      None => None,
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    self.source_iter.size_hint()
  }
}

impl<I, F, T1, T2, T> Batch<T2, T> for ProjectionIterator<I, F, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  I: Batch<T1, T>,
  F: Fn(T1) -> T2 + Clone,
{
}
