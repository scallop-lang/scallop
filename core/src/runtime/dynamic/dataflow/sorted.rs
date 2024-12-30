use super::*;

#[derive(Clone)]
pub struct DynamicSortedDataflow<'a, Prov: Provenance> {
  pub source: DynamicDataflow<'a, Prov>,
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicSortedDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    self.source.iter_stable().unary(SortOp)
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    self.source.iter_recent().unary(SortOp)
  }
}

#[derive(Clone)]
pub struct SortOp;

impl<'a, Prov: Provenance> BatchUnaryOp<'a, Prov> for SortOp {
  fn apply(&self, b: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    DynamicBatch::new(DynamicSortedBatch::new(b))
  }
}

#[derive(Clone)]
pub struct DynamicSortedBatch<'a, Prov: Provenance> {
  src: DynamicBatch<'a, Prov>,
  sorted_tgt: Option<std::vec::IntoIter<DynamicElement<Prov>>>,
}

impl<'a, Prov: Provenance> DynamicSortedBatch<'a, Prov> {
  pub fn new(src: DynamicBatch<'a, Prov>) -> Self {
    Self {
      src,
      sorted_tgt: None,
    }
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicSortedBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    if let Some(iterator) = &mut self.sorted_tgt {
      iterator.next()
    } else {
      let mut tgt = self.src.collect_vec();
      tgt.sort();
      let mut sorted_tgt_iter = tgt.into_iter();
      let maybe_first_elem = sorted_tgt_iter.next();
      if let Some(first_elem) = maybe_first_elem {
        self.sorted_tgt = Some(sorted_tgt_iter);
        Some(first_elem)
      } else {
        None
      }
    }
  }
}
