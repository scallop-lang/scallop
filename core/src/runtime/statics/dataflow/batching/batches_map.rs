use std::marker::PhantomData;

use super::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

#[derive(Clone)]
pub struct BatchesMap<B, Op, T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
  Op: BatchUnaryOp<B::Batch>,
  Op::I2: Batch<T2, Prov>,
  B: Batches<T1, Prov>,
{
  source: B,
  op: Op,
  phantom: PhantomData<(T1, T2, Prov)>,
}

impl<B, Op, T1, T2, Prov> BatchesMap<B, Op, T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
  Op: BatchUnaryOp<B::Batch>,
  Op::I2: Batch<T2, Prov>,
  B: Batches<T1, Prov>,
{
  pub fn new(source: B, op: Op) -> Self {
    Self {
      source,
      op,
      phantom: PhantomData,
    }
  }
}

impl<B, Op, T1, T2, Prov> Iterator for BatchesMap<B, Op, T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
  Op: BatchUnaryOp<B::Batch>,
  Op::I2: Batch<T2, Prov>,
  B: Batches<T1, Prov>,
{
  type Item = Op::I2;

  fn next(&mut self) -> Option<Self::Item> {
    self.source.next().map(|batch| self.op.apply(batch))
  }
}

impl<B, Op, T1, T2, Prov> Batches<T2, Prov> for BatchesMap<B, Op, T1, T2, Prov>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  Prov: Provenance,
  Op: BatchUnaryOp<B::Batch>,
  Op::I2: Batch<T2, Prov>,
  B: Batches<T1, Prov>,
{
  type Batch = Op::I2;
}
