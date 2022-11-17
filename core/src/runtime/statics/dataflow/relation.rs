use std::cell::Ref;

use super::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

impl<'a, Tup, Prov> Dataflow<Tup, Prov> for &'a StaticRelation<Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  type Stable = StableRelationBatches<'a, Tup, Prov>;

  type Recent = SingleBatch<RelationIterator<'a, Tup, Prov>>;

  fn iter_stable(&self) -> Self::Stable {
    Self::Stable {
      relations: self.stable.borrow(),
      rela_id: 0,
    }
  }

  fn iter_recent(self) -> Self::Recent {
    Self::Recent::singleton(RelationIterator {
      relation: self.recent.borrow(),
      elem_id: 0,
    })
  }
}

pub struct StableRelationBatches<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  relations: Ref<'a, Vec<StaticCollection<Tup, Prov>>>,
  rela_id: usize,
}

impl<'a, Tup, Prov> Clone for StableRelationBatches<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self {
      relations: Ref::clone(&self.relations),
      rela_id: self.rela_id.clone(),
    }
  }
}

impl<'a, Tup, Prov> Iterator for StableRelationBatches<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  type Item = StableRelationBatch<'a, Tup, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.rela_id < self.relations.len() {
      let result = Self::Item {
        relations: Ref::clone(&self.relations),
        rela_id: self.rela_id,
        elem_id: 0,
      };
      self.rela_id += 1;
      return Some(result);
    } else {
      return None;
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let size = self.relations.len();
    (size, Some(size))
  }
}

impl<'a, Tup, Prov> Batches<Tup, Prov> for StableRelationBatches<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  type Batch = StableRelationBatch<'a, Tup, Prov>;
}

pub struct StableRelationBatch<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  relations: Ref<'a, Vec<StaticCollection<Tup, Prov>>>,
  rela_id: usize,
  elem_id: usize,
}

impl<'a, Tup, Prov> Clone for StableRelationBatch<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self {
      relations: Ref::clone(&self.relations),
      rela_id: self.rela_id.clone(),
      elem_id: self.elem_id.clone(),
    }
  }
}

impl<'a, Tup, Prov> Iterator for StableRelationBatch<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  type Item = StaticElement<Tup, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    let relation = &self.relations[self.rela_id];
    if self.elem_id < relation.len() {
      let elem = &relation[self.elem_id];
      self.elem_id += 1;
      return Some(elem.clone());
    } else {
      return None;
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let size = self.relations[self.rela_id].len();
    (size, Some(size))
  }
}

impl<'a, Tup, Prov> Batch<Tup, Prov> for StableRelationBatch<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
}

pub struct RelationIterator<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  relation: Ref<'a, StaticCollection<Tup, Prov>>,
  elem_id: usize,
}

impl<'a, Tup, Prov> Clone for RelationIterator<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self {
      relation: Ref::clone(&self.relation),
      elem_id: self.elem_id,
    }
  }
}

impl<'a, Tup, Prov> Iterator for RelationIterator<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  type Item = StaticElement<Tup, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.elem_id < self.relation.len() {
      let elem = &self.relation[self.elem_id];
      self.elem_id += 1;
      return Some(elem.clone());
    } else {
      return None;
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let size = self.relation.len();
    (size, Some(size))
  }
}

impl<'a, Tup, Prov> Batch<Tup, Prov> for RelationIterator<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  fn step(&mut self, u: usize) {
    self.elem_id += u;
  }

  fn search_ahead<F>(&mut self, mut cmp: F) -> Option<StaticElement<Tup, Prov>>
  where
    F: FnMut(&Tup) -> bool,
  {
    assert!(self.elem_id > 0);
    let mut curr = self.elem_id - 1;
    if curr < self.relation.len() && cmp(&self.relation[curr].tuple) {
      let mut step = 1;
      while curr + step < self.relation.len() && cmp(&self.relation[curr + step].tuple) {
        curr += step;
        step <<= 1;
      }

      step >>= 1;
      while step > 0 {
        if curr + step < self.relation.len() && cmp(&self.relation[curr + step].tuple) {
          curr += step;
        }
        step >>= 1;
      }
      self.elem_id = curr + 1;
      self.next()
    } else {
      None
    }
  }
}
