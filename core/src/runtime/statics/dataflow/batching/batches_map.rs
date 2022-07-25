use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

#[derive(Clone)]
pub struct BatchesMap<B, Op, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  Op: BatchUnaryOp<B::Batch>,
  Op::I2: Batch<T2, T>,
  B: Batches<T1, T>,
{
  source: B,
  op: Op,
  phantom: PhantomData<(T1, T2, T)>,
}

impl<B, Op, T1, T2, T> BatchesMap<B, Op, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  Op: BatchUnaryOp<B::Batch>,
  Op::I2: Batch<T2, T>,
  B: Batches<T1, T>,
{
  pub fn new(source: B, op: Op) -> Self {
    Self {
      source,
      op,
      phantom: PhantomData,
    }
  }
}

impl<B, Op, T1, T2, T> Iterator for BatchesMap<B, Op, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  Op: BatchUnaryOp<B::Batch>,
  Op::I2: Batch<T2, T>,
  B: Batches<T1, T>,
{
  type Item = Op::I2;

  fn next(&mut self) -> Option<Self::Item> {
    self.source.next().map(|batch| self.op.apply(batch))
  }
}

impl<B, Op, T1, T2, T> Batches<T2, T> for BatchesMap<B, Op, T1, T2, T>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  T: Tag,
  Op: BatchUnaryOp<B::Batch>,
  Op::I2: Batch<T2, T>,
  B: Batches<T1, T>,
{
  type Batch = Op::I2;
}
