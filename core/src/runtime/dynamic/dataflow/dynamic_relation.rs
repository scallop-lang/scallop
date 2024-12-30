use std::cell::Ref;

use crate::common::tuple::*;

use super::batching::*;
use super::*;

#[derive(Clone)]
pub struct DynamicRelationDataflow<'a, Prov: Provenance>(pub &'a DynamicRelation<Prov>);

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicRelationDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::new(DynamicRelationStableBatches {
      collections: self.0.stable.borrow(),
      rela_id: 0,
    })
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    let b = DynamicRelationRecentBatch {
      collection: self.0.recent.borrow(),
      elem_id: 0,
    };
    DynamicBatches::single(b)
  }
}

pub struct DynamicRelationStableBatches<'a, Prov: Provenance> {
  pub collections: Ref<'a, Vec<DynamicSortedCollection<Prov>>>,
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

impl<'a, Prov: Provenance> Batches<'a, Prov> for DynamicRelationStableBatches<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    if self.rela_id < self.collections.len() {
      let result = DynamicRelationStableBatch {
        collections: Ref::clone(&self.collections),
        rela_id: self.rela_id,
        elem_id: 0,
      };
      self.rela_id += 1;
      Some(DynamicBatch::new(result))
    } else {
      None
    }
  }
}

pub struct DynamicRelationStableBatch<'a, Prov: Provenance> {
  pub collections: Ref<'a, Vec<DynamicSortedCollection<Prov>>>,
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

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicRelationStableBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    let relation = &self.collections[self.rela_id];
    if self.elem_id < relation.len() {
      let elem = &relation[self.elem_id];
      self.elem_id += 1;
      Some(elem.clone())
    } else {
      None
    }
  }

  fn step(&mut self, u: usize) {
    self.elem_id += u;
  }

  fn search_until(&mut self, until: &Tuple) -> Option<DynamicElement<Prov>> {
    let col = &self.collections[self.rela_id];
    if search_ahead_variable_helper(col, &mut self.elem_id, |t| t < until) {
      self.next_elem()
    } else {
      None
    }
  }

  fn search_elem_0_until(&mut self, until: &Tuple) -> Option<DynamicElement<Prov>> {
    let col = &self.collections[self.rela_id];
    if search_ahead_variable_helper(col, &mut self.elem_id, |t| &t[0] < until) {
      self.next_elem()
    } else {
      None
    }
  }
}

pub struct DynamicRelationRecentBatch<'a, Prov: Provenance> {
  pub collection: Ref<'a, DynamicSortedCollection<Prov>>,
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

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicRelationRecentBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    if self.elem_id < self.collection.len() {
      let elem = &self.collection[self.elem_id];
      self.elem_id += 1;
      Some(elem.clone())
    } else {
      None
    }
  }

  fn step(&mut self, u: usize) {
    self.elem_id += u;
  }

  fn search_until(&mut self, until: &Tuple) -> Option<DynamicElement<Prov>> {
    if search_ahead_variable_helper(&self.collection, &mut self.elem_id, |t| t < until) {
      self.next_elem()
    } else {
      None
    }
  }

  fn search_elem_0_until(&mut self, until: &Tuple) -> Option<DynamicElement<Prov>> {
    if search_ahead_variable_helper(&self.collection, &mut self.elem_id, |t| &t[0] < until) {
      self.next_elem()
    } else {
      None
    }
  }
}

fn search_ahead_variable_helper<Prov, F>(collection: &DynamicSortedCollection<Prov>, elem_id: &mut usize, mut cmp: F) -> bool
where
  Prov: Provenance,
  F: FnMut(&Tuple) -> bool,
{
  assert!(*elem_id > 0);
  let mut curr = *elem_id - 1;
  if curr < collection.len() && cmp(&collection[curr].tuple) {
    let mut step = 1;
    while curr + step < collection.len() && cmp(&collection[curr + step].tuple) {
      curr += step;
      step <<= 1;
    }
    step >>= 1;
    while step > 0 {
      if curr + step < collection.len() && cmp(&collection[curr + step].tuple) {
        curr += step;
      }
      step >>= 1;
    }
    *elem_id = curr + 1;
    true
  } else {
    false
  }
}
