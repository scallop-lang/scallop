use std::cell::Ref;

use super::batching::*;
use super::*;

#[derive(Clone)]
pub struct DynamicRelationDataflow<'a, Prov: Provenance>(pub &'a DynamicRelation<Prov>);

impl<'a, Prov: Provenance> DynamicRelationDataflow<'a, Prov> {
  pub fn iter_stable(&self, _: &RuntimeEnvironment) -> DynamicBatches<'a, Prov> {
    DynamicBatches::DynamicRelationStable(DynamicRelationStableBatches {
      collections: self.0.stable.borrow(),
      rela_id: 0,
    })
  }

  pub fn iter_recent(&self, _: &RuntimeEnvironment) -> DynamicBatches<'a, Prov> {
    let b = DynamicRelationRecentBatch {
      collection: self.0.recent.borrow(),
      elem_id: 0,
    };
    DynamicBatches::single(DynamicBatch::DynamicRelationRecent(b))
  }
}

pub struct DynamicRelationStableBatches<'a, Prov: Provenance> {
  pub collections: Ref<'a, Vec<DynamicCollection<Prov>>>,
  pub rela_id: usize,
}

impl<'a, Prov: Provenance> Clone for DynamicRelationStableBatches<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      collections: Ref::clone(&self.collections),
      rela_id: self.rela_id,
    }
  }
}

impl<'a, Prov: Provenance> Iterator for DynamicRelationStableBatches<'a, Prov> {
  type Item = DynamicBatch<'a, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.rela_id < self.collections.len() {
      let result = DynamicRelationStableBatch {
        collections: Ref::clone(&self.collections),
        rela_id: self.rela_id,
        elem_id: 0,
      };
      self.rela_id += 1;
      Some(DynamicBatch::DynamicRelationStable(result))
    } else {
      None
    }
  }
}

pub struct DynamicRelationStableBatch<'a, Prov: Provenance> {
  pub collections: Ref<'a, Vec<DynamicCollection<Prov>>>,
  pub rela_id: usize,
  pub elem_id: usize,
}

impl<'a, Prov: Provenance> Clone for DynamicRelationStableBatch<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      collections: Ref::clone(&self.collections),
      rela_id: self.rela_id,
      elem_id: self.elem_id,
    }
  }
}

impl<'a, Prov: Provenance> Iterator for DynamicRelationStableBatch<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    let relation = &self.collections[self.rela_id];
    if self.elem_id < relation.len() {
      let elem = &relation[self.elem_id];
      self.elem_id += 1;
      Some(elem.clone())
    } else {
      None
    }
  }
}

pub struct DynamicRelationRecentBatch<'a, Prov: Provenance> {
  pub collection: Ref<'a, DynamicCollection<Prov>>,
  pub elem_id: usize,
}

impl<'a, Prov: Provenance> Clone for DynamicRelationRecentBatch<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      collection: Ref::clone(&self.collection),
      elem_id: self.elem_id,
    }
  }
}

impl<'a, Prov: Provenance> Iterator for DynamicRelationRecentBatch<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.elem_id < self.collection.len() {
      let elem = &self.collection[self.elem_id];
      self.elem_id += 1;
      Some(elem.clone())
    } else {
      None
    }
  }
}
