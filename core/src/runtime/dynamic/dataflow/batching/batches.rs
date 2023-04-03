use super::super::*;
use super::*;
use crate::common::expr::Expr;
use crate::common::tuple::Tuple;
use crate::runtime::provenance::*;

#[derive(Clone)]
pub enum DynamicBatches<'a, Prov: Provenance> {
  Empty,
  Single(Option<DynamicBatch<'a, Prov>>),
  Optional(DynamicBatchesOptional<'a, Prov>),
  Chain(DynamicBatchesChain<'a, Prov>),
  DynamicRelationStable(DynamicRelationStableBatches<'a, Prov>),
  Project(DynamicProjectBatches<'a, Prov>),
  Filter(DynamicFilterBatches<'a, Prov>),
  Find(DynamicFindBatches<'a, Prov>),
  OverwriteOne(DynamicOverwriteOneBatches<'a, Prov>),
  Binary(DynamicBatchesBinary<'a, Prov>),
  ForeignPredicateConstraint(ForeignPredicateConstraintBatches<'a, Prov>),
  ForeignPredicateJoin(ForeignPredicateJoinBatches<'a, Prov>),
}

impl<'a, Prov: Provenance> DynamicBatches<'a, Prov> {
  pub fn empty() -> Self {
    Self::Empty
  }

  pub fn single(batch: DynamicBatch<'a, Prov>) -> Self {
    Self::Single(Some(batch))
  }

  pub fn chain(bs: Vec<DynamicBatches<'a, Prov>>) -> Self {
    if bs.is_empty() {
      Self::Empty
    } else {
      Self::Chain(DynamicBatchesChain { bs, id: 0 })
    }
  }

  pub fn project(runtime: &'a RuntimeEnvironment, source: DynamicBatches<'a, Prov>, expression: Expr) -> Self {
    Self::Project(DynamicProjectBatches {
      runtime,
      source: Box::new(source),
      expression,
    })
  }

  pub fn filter(runtime: &'a RuntimeEnvironment, source: DynamicBatches<'a, Prov>, filter: Expr) -> Self {
    Self::Filter(DynamicFilterBatches {
      runtime,
      source: Box::new(source),
      filter,
    })
  }

  pub fn find(source: DynamicBatches<'a, Prov>, key: Tuple) -> Self {
    Self::Find(DynamicFindBatches {
      source: Box::new(source),
      key,
    })
  }

  pub fn binary(b1: DynamicBatches<'a, Prov>, b2: DynamicBatches<'a, Prov>, op: BatchBinaryOp<'a, Prov>) -> Self {
    Self::Binary(DynamicBatchesBinary::new(b1, b2, op))
  }
}

impl<'a, Prov: Provenance> Iterator for DynamicBatches<'a, Prov> {
  type Item = DynamicBatch<'a, Prov>;

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
      Self::OverwriteOne(o) => o.next(),
      Self::Binary(b) => b.next(),
      Self::ForeignPredicateConstraint(b) => b.next(),
      Self::ForeignPredicateJoin(b) => b.next(),
    }
  }
}

#[derive(Clone)]
pub struct DynamicBatchesOptional<'a, Prov: Provenance> {
  optional_batches: Option<Box<DynamicBatches<'a, Prov>>>,
}

impl<'a, Prov: Provenance> Iterator for DynamicBatchesOptional<'a, Prov> {
  type Item = DynamicBatch<'a, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    match &mut self.optional_batches {
      Some(b) => b.next(),
      None => None,
    }
  }
}

#[derive(Clone)]
pub struct DynamicBatchesChain<'a, Prov: Provenance> {
  bs: Vec<DynamicBatches<'a, Prov>>,
  id: usize,
}

impl<'a, Prov: Provenance> Iterator for DynamicBatchesChain<'a, Prov> {
  type Item = DynamicBatch<'a, Prov>;

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
pub struct DynamicBatchesBinary<'a, Prov: Provenance> {
  b1: Box<DynamicBatches<'a, Prov>>,
  b1_curr: Option<DynamicBatch<'a, Prov>>,
  b2: Box<DynamicBatches<'a, Prov>>,
  b2_source: Box<DynamicBatches<'a, Prov>>,
  op: BatchBinaryOp<'a, Prov>,
}

impl<'a, Prov: Provenance> DynamicBatchesBinary<'a, Prov> {
  pub fn new(mut b1: DynamicBatches<'a, Prov>, b2: DynamicBatches<'a, Prov>, op: BatchBinaryOp<'a, Prov>) -> Self {
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

impl<'a, Prov: Provenance> Iterator for DynamicBatchesBinary<'a, Prov> {
  type Item = DynamicBatch<'a, Prov>;

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
