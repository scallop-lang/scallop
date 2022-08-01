use super::super::*;
use super::*;
use crate::common::expr::Expr;
use crate::common::tuple::Tuple;
use crate::runtime::provenance::*;

#[derive(Clone)]
pub enum DynamicBatches<'a, T: Tag> {
  Empty,
  Single(Option<DynamicBatch<'a, T>>),
  Optional(DynamicBatchesOptional<'a, T>),
  Chain(DynamicBatchesChain<'a, T>),
  DynamicRelationStable(DynamicRelationStableBatches<'a, T>),
  Project(DynamicProjectBatches<'a, T>),
  Filter(DynamicFilterBatches<'a, T>),
  Find(DynamicFindBatches<'a, T>),
  Binary(DynamicBatchesBinary<'a, T>),
}

impl<'a, T: Tag> DynamicBatches<'a, T> {
  pub fn empty() -> Self {
    Self::Empty
  }

  pub fn single(batch: DynamicBatch<'a, T>) -> Self {
    Self::Single(Some(batch))
  }

  pub fn chain(bs: Vec<DynamicBatches<'a, T>>) -> Self {
    if bs.is_empty() {
      Self::Empty
    } else {
      Self::Chain(DynamicBatchesChain { bs, id: 0 })
    }
  }

  pub fn project(source: DynamicBatches<'a, T>, expression: Expr) -> Self {
    Self::Project(DynamicProjectBatches {
      source: Box::new(source),
      expression,
    })
  }

  pub fn filter(source: DynamicBatches<'a, T>, filter: Expr) -> Self {
    Self::Filter(DynamicFilterBatches {
      source: Box::new(source),
      filter,
    })
  }

  pub fn find(source: DynamicBatches<'a, T>, key: Tuple) -> Self {
    Self::Find(DynamicFindBatches {
      source: Box::new(source),
      key,
    })
  }

  pub fn binary(b1: DynamicBatches<'a, T>, b2: DynamicBatches<'a, T>, op: BatchBinaryOp<'a, T>) -> Self {
    Self::Binary(DynamicBatchesBinary::new(b1, b2, op))
  }
}

impl<'a, T: Tag> Iterator for DynamicBatches<'a, T> {
  type Item = DynamicBatch<'a, T>;

  fn next(&mut self) -> Option<Self::Item> {
    match self {
      Self::Empty => None,
      Self::Single(s) => s.take(),
      Self::Optional(o) => o.next(),
      Self::Chain(c) => c.next(),
      Self::DynamicRelationStable(drs) => drs.next(),
      Self::Project(m) => m.next(),
      Self::Filter(f) => f.next(),
      Self::Find(f) => f.next(),
      Self::Binary(b) => b.next(),
    }
  }
}

#[derive(Clone)]
pub struct DynamicBatchesOptional<'a, T: Tag> {
  optional_batches: Option<Box<DynamicBatches<'a, T>>>,
}

impl<'a, T: Tag> Iterator for DynamicBatchesOptional<'a, T> {
  type Item = DynamicBatch<'a, T>;

  fn next(&mut self) -> Option<Self::Item> {
    match &mut self.optional_batches {
      Some(b) => b.next(),
      None => None,
    }
  }
}

#[derive(Clone)]
pub struct DynamicBatchesChain<'a, T: Tag> {
  bs: Vec<DynamicBatches<'a, T>>,
  id: usize,
}

impl<'a, T: Tag> Iterator for DynamicBatchesChain<'a, T> {
  type Item = DynamicBatch<'a, T>;

  fn next(&mut self) -> Option<Self::Item> {
    while self.id < self.bs.len() {
      if let Some(batch) = self.bs[self.id].next() {
        return Some(batch);
      } else {
        self.id += 1;
      }
    }
    None
  }
}

#[derive(Clone)]
pub struct DynamicBatchesBinary<'a, T: Tag> {
  b1: Box<DynamicBatches<'a, T>>,
  b1_curr: Option<DynamicBatch<'a, T>>,
  b2: Box<DynamicBatches<'a, T>>,
  b2_source: Box<DynamicBatches<'a, T>>,
  op: BatchBinaryOp<'a, T>,
}

impl<'a, T: Tag> DynamicBatchesBinary<'a, T> {
  pub fn new(mut b1: DynamicBatches<'a, T>, b2: DynamicBatches<'a, T>, op: BatchBinaryOp<'a, T>) -> Self {
    let b1_curr = b1.next();
    let b2_source = b2.clone();
    Self {
      b1: Box::new(b1),
      b1_curr,
      b2: Box::new(b2),
      b2_source: Box::new(b2_source),
      op,
    }
  }
}

impl<'a, T: Tag> Iterator for DynamicBatchesBinary<'a, T> {
  type Item = DynamicBatch<'a, T>;

  fn next(&mut self) -> Option<Self::Item> {
    loop {
      match &self.b1_curr {
        Some(b1_curr_batch) => match self.b2.next() {
          Some(b2_curr_batch) => {
            let result = self.op.apply(b1_curr_batch.clone(), b2_curr_batch);
            return Some(result);
          }
          None => {
            self.b1_curr = self.b1.next();
            self.b2 = self.b2_source.clone();
          }
        },
        None => return None,
      }
    }
  }
}
