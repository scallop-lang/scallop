use crate::common::tuple::Tuple;
use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

use super::super::*;

#[derive(Clone)]
pub enum DynamicBatch<'a, Prov: Provenance> {
  Vec(std::slice::Iter<'a, DynamicElement<Prov>>),
  SourceVec(std::vec::IntoIter<DynamicElement<Prov>>),
  DynamicRelationStable(DynamicRelationStableBatch<'a, Prov>),
  DynamicRelationRecent(DynamicRelationRecentBatch<'a, Prov>),
  OverwriteOne(DynamicOverwriteOneBatch<'a, Prov>),
  Project(DynamicProjectBatch<'a, Prov>),
  Filter(DynamicFilterBatch<'a, Prov>),
  Find(DynamicFindBatch<'a, Prov>),
  Intersect(DynamicIntersectBatch<'a, Prov>),
  Join(DynamicJoinBatch<'a, Prov>),
  Product(DynamicProductBatch<'a, Prov>),
  Difference(DynamicDifferenceBatch<'a, Prov>),
  Antijoin(DynamicAntijoinBatch<'a, Prov>),
  ForeignPredicateConstraint(ForeignPredicateConstraintBatch<'a, Prov>),
  ForeignPredicateJoin(ForeignPredicateJoinBatch<'a, Prov>),
}

impl<'a, Prov: Provenance> DynamicBatch<'a, Prov> {
  pub fn vec(v: &'a Vec<DynamicElement<Prov>>) -> Self {
    Self::Vec(v.iter())
  }

  pub fn source_vec(v: Vec<DynamicElement<Prov>>) -> Self {
    Self::SourceVec(v.into_iter())
  }

  pub fn step(&mut self, u: usize) {
    match self {
      Self::DynamicRelationStable(s) => s.elem_id += u,
      Self::DynamicRelationRecent(r) => r.elem_id += u,
      _ => {
        for _ in 0..u {
          self.next();
        }
      }
    }
  }

  pub fn search_ahead<F>(&mut self, cmp: F) -> Option<DynamicElement<Prov>>
  where
    F: FnMut(&Tuple) -> bool,
  {
    fn search_ahead_variable_helper_1<Prov, F>(
      collection: &DynamicCollection<Prov>,
      elem_id: &mut usize,
      mut cmp: F,
    ) -> bool
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

    match self {
      Self::DynamicRelationStable(s) => {
        let col = &s.collections[s.rela_id];
        if search_ahead_variable_helper_1(col, &mut s.elem_id, cmp) {
          self.next()
        } else {
          None
        }
      }
      Self::DynamicRelationRecent(r) => {
        if search_ahead_variable_helper_1(&r.collection, &mut r.elem_id, cmp) {
          self.next()
        } else {
          None
        }
      }
      _ => self.next(),
    }
  }
}

impl<'a, Prov: Provenance> Iterator for DynamicBatch<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    match self {
      Self::Vec(iter) => iter.next().map(Clone::clone),
      Self::SourceVec(iter) => iter.next(),
      Self::DynamicRelationStable(b) => b.next(),
      Self::DynamicRelationRecent(b) => b.next(),
      Self::OverwriteOne(o) => o.next(),
      Self::Project(p) => p.next(),
      Self::Filter(f) => f.next(),
      Self::Find(f) => f.next(),
      Self::Intersect(i) => i.next(),
      Self::Join(j) => j.next(),
      Self::Product(p) => p.next(),
      Self::Difference(d) => d.next(),
      Self::Antijoin(a) => a.next(),
      Self::ForeignPredicateConstraint(b) => b.next(),
      Self::ForeignPredicateJoin(b) => b.next(),
    }
  }
}
