use std::cell::Ref;

use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

use super::batching::*;

#[derive(Clone)]
pub struct DynamicRelationDataflow<'a, T: Tag>(pub &'a DynamicRelation<T>);

impl<'a, T: Tag> DynamicRelationDataflow<'a, T> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::DynamicRelationStable(DynamicRelationStableBatches {
      collections: self.0.stable.borrow(),
      rela_id: 0,
    })
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    let b = DynamicRelationRecentBatch {
      collection: self.0.recent.borrow(),
      elem_id: 0,
    };
    DynamicBatches::single(DynamicBatch::DynamicRelationRecent(b))
  }
}

pub struct DynamicRelationStableBatches<'a, T: Tag> {
  pub collections: Ref<'a, Vec<DynamicCollection<T>>>,
  pub rela_id: usize,
}

impl<'a, T: Tag> Clone for DynamicRelationStableBatches<'a, T> {
  fn clone(&self) -> Self {
    Self {
      collections: Ref::clone(&self.collections),
      rela_id: self.rela_id,
    }
  }
}

impl<'a, T: Tag> Iterator for DynamicRelationStableBatches<'a, T> {
  type Item = DynamicBatch<'a, T>;

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

pub struct DynamicRelationStableBatch<'a, T: Tag> {
  pub collections: Ref<'a, Vec<DynamicCollection<T>>>,
  pub rela_id: usize,
  pub elem_id: usize,
}

impl<'a, T: Tag> Clone for DynamicRelationStableBatch<'a, T> {
  fn clone(&self) -> Self {
    Self {
      collections: Ref::clone(&self.collections),
      rela_id: self.rela_id,
      elem_id: self.elem_id,
    }
  }
}

impl<'a, T: Tag> Iterator for DynamicRelationStableBatch<'a, T> {
  type Item = DynamicElement<T>;

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

pub struct DynamicRelationRecentBatch<'a, T: Tag> {
  pub collection: Ref<'a, DynamicCollection<T>>,
  pub elem_id: usize,
}

impl<'a, T: Tag> Clone for DynamicRelationRecentBatch<'a, T> {
  fn clone(&self) -> Self {
    Self {
      collection: Ref::clone(&self.collection),
      elem_id: self.elem_id,
    }
  }
}

impl<'a, T: Tag> Iterator for DynamicRelationRecentBatch<'a, T> {
  type Item = DynamicElement<T>;

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
