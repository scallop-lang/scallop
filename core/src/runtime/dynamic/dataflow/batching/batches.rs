use super::*;
use crate::runtime::provenance::*;

pub trait Batches<'a, Prov: Provenance>: dyn_clone::DynClone + 'a {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>>;
}

pub struct DynamicBatches<'a, Prov: Provenance>(Box<dyn Batches<'a, Prov>>);

impl<'a, Prov: Provenance> Batches<'a, Prov> for DynamicBatches<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    self.0.next_batch()
  }
}

impl<'a, Prov: Provenance> DynamicBatches<'a, Prov> {
  pub fn new<B: Batches<'a, Prov>>(b: B) -> Self {
    Self(Box::new(b))
  }

  pub fn empty() -> Self {
    Self::new(EmptyBatches)
  }

  pub fn single<B: Batch<'a, Prov>>(b: B) -> Self {
    Self::new(SingleBatch::new(b))
  }

  pub fn chain(bs: Vec<DynamicBatches<'a, Prov>>) -> Self {
    Self::new(DynamicBatchesChain::new(bs))
  }

  pub fn binary<Op: BatchBinaryOp<'a, Prov>>(b1: Self, b2: Self, op: Op) -> Self {
    Self::new(DynamicBatchesBinary::new(b1, b2, op))
  }
}

impl<'a, Prov: Provenance> Clone for DynamicBatches<'a, Prov> {
  fn clone(&self) -> Self {
    Self(dyn_clone::clone_box(&*self.0))
  }
}

impl<'a, Prov: Provenance> Iterator for DynamicBatches<'a, Prov> {
  type Item = DynamicBatch<'a, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    self.next_batch()
  }
}

#[derive(Clone)]
pub struct EmptyBatches;

impl<'a, Prov: Provenance> Batches<'a, Prov> for EmptyBatches {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    None
  }
}

#[derive(Clone)]
pub struct SingleBatch<'a, Prov: Provenance>(Option<DynamicBatch<'a, Prov>>);

impl<'a, Prov: Provenance> SingleBatch<'a, Prov> {
  pub fn new<B: Batch<'a, Prov>>(b: B) -> Self {
    Self(Some(DynamicBatch::new(b)))
  }
}

impl<'a, Prov: Provenance> Batches<'a, Prov> for SingleBatch<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    self.0.take()
  }
}

#[allow(unused)]
#[derive(Clone)]
pub struct DynamicBatchesOptional<'a, Prov: Provenance> {
  optional_batches: Option<DynamicBatches<'a, Prov>>,
}

impl<'a, Prov: Provenance> Batches<'a, Prov> for DynamicBatchesOptional<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    match &mut self.optional_batches {
      Some(b) => b.next_batch(),
      None => None,
    }
  }
}

#[derive(Clone)]
pub struct DynamicBatchesChain<'a, Prov: Provenance> {
  bs: Vec<DynamicBatches<'a, Prov>>,
  id: usize,
}

impl<'a, Prov: Provenance> DynamicBatchesChain<'a, Prov> {
  pub fn new(bs: Vec<DynamicBatches<'a, Prov>>) -> Self {
    Self { bs, id: 0 }
  }
}

impl<'a, Prov: Provenance> Batches<'a, Prov> for DynamicBatchesChain<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    while self.id < self.bs.len() {
      if let Some(batch) = self.bs[self.id].next_batch() {
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
  b1: DynamicBatches<'a, Prov>,
  b1_curr: Option<DynamicBatch<'a, Prov>>,
  b2: DynamicBatches<'a, Prov>,
  b2_source: DynamicBatches<'a, Prov>,
  op: DynamicBatchBinaryOp<'a, Prov>,
}

impl<'a, Prov: Provenance> DynamicBatchesBinary<'a, Prov> {
  pub fn new<Op: BatchBinaryOp<'a, Prov>>(
    mut b1: DynamicBatches<'a, Prov>,
    b2: DynamicBatches<'a, Prov>,
    op: Op,
  ) -> Self {
    let b1_curr = b1.next_batch();
    let b2_source = b2.clone();
    Self {
      b1: b1,
      b1_curr,
      b2: b2,
      b2_source: b2_source,
      op: DynamicBatchBinaryOp::new(op),
    }
  }
}

impl<'a, Prov: Provenance> Batches<'a, Prov> for DynamicBatchesBinary<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    loop {
      match &self.b1_curr {
        Some(b1_curr_batch) => match self.b2.next_batch() {
          Some(b2_curr_batch) => {
            let result = self.op.apply(b1_curr_batch.clone(), b2_curr_batch);
            return Some(result);
          }
          None => {
            self.b1_curr = self.b1.next_batch();
            self.b2 = self.b2_source.clone();
          }
        },
        None => return None,
      }
    }
  }
}
