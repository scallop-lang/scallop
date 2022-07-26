use crate::common::expr::Expr;
use crate::common::tuple::Tuple;
use crate::common::tuple_type::TupleType;
use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub enum DynamicDataflow<'a, Prov: Provenance> {
  StableUnit(DynamicStableUnitDataflow<'a, Prov>),
  RecentUnit(DynamicRecentUnitDataflow<'a, Prov>),
  Vec(&'a Vec<DynamicElement<Prov>>),
  DynamicStableCollection(DynamicStableCollectionDataflow<'a, Prov>),
  DynamicRecentCollection(DynamicRecentCollectionDataflow<'a, Prov>),
  DynamicRelation(DynamicRelationDataflow<'a, Prov>),
  OverwriteOne(DynamicOverwriteOneDataflow<'a, Prov>),
  Project(DynamicProjectDataflow<'a, Prov>),
  Filter(DynamicFilterDataflow<'a, Prov>),
  Find(DynamicFindDataflow<'a, Prov>),
  Intersect(DynamicIntersectDataflow<'a, Prov>),
  Join(DynamicJoinDataflow<'a, Prov>),
  Product(DynamicProductDataflow<'a, Prov>),
  Union(DynamicUnionDataflow<'a, Prov>),
  Difference(DynamicDifferenceDataflow<'a, Prov>),
  Antijoin(DynamicAntijoinDataflow<'a, Prov>),
  Aggregate(DynamicAggregationDataflow<'a, Prov>),
}

impl<'a, Prov: Provenance> DynamicDataflow<'a, Prov> {
  pub fn vec(vec: &'a Vec<DynamicElement<Prov>>) -> Self {
    Self::Vec(vec)
  }

  pub fn recent_unit(ctx: &'a Prov, tuple_type: TupleType) -> Self {
    Self::RecentUnit(DynamicRecentUnitDataflow::new(ctx, tuple_type))
  }

  pub fn stable_unit(ctx: &'a Prov, tuple_type: TupleType) -> Self {
    Self::StableUnit(DynamicStableUnitDataflow::new(ctx, tuple_type))
  }

  pub fn dynamic_collection(col: &'a DynamicCollection<Prov>, recent: bool) -> Self {
    if recent {
      Self::dynamic_recent_collection(col)
    } else {
      Self::dynamic_stable_collection(col)
    }
  }

  pub fn dynamic_stable_collection(col: &'a DynamicCollection<Prov>) -> Self {
    Self::DynamicStableCollection(DynamicStableCollectionDataflow(col))
  }

  pub fn dynamic_recent_collection(col: &'a DynamicCollection<Prov>) -> Self {
    Self::DynamicRecentCollection(DynamicRecentCollectionDataflow(col))
  }

  pub fn dynamic_relation(rela: &'a DynamicRelation<Prov>) -> Self {
    Self::DynamicRelation(DynamicRelationDataflow(rela))
  }

  pub fn overwrite_one(self, ctx: &'a Prov) -> Self {
    Self::OverwriteOne(DynamicOverwriteOneDataflow {
      source: Box::new(self),
      ctx,
    })
  }

  pub fn project(self, expression: Expr) -> Self {
    Self::Project(DynamicProjectDataflow {
      source: Box::new(self),
      expression,
    })
  }

  pub fn filter(self, filter: Expr) -> Self {
    Self::Filter(DynamicFilterDataflow {
      source: Box::new(self),
      filter,
    })
  }

  pub fn find(self, key: Tuple) -> Self {
    Self::Find(DynamicFindDataflow {
      source: Box::new(self),
      key,
    })
  }

  pub fn intersect(self, d2: Self, ctx: &'a Prov) -> Self {
    Self::Intersect(DynamicIntersectDataflow {
      d1: Box::new(self),
      d2: Box::new(d2),
      ctx,
    })
  }

  pub fn join(self, d2: Self, ctx: &'a Prov) -> Self {
    Self::Join(DynamicJoinDataflow {
      d1: Box::new(self),
      d2: Box::new(d2),
      ctx,
    })
  }

  pub fn product(self, d2: Self, ctx: &'a Prov) -> Self {
    Self::Product(DynamicProductDataflow {
      d1: Box::new(self),
      d2: Box::new(d2),
      ctx,
    })
  }

  pub fn union(self, d2: Self) -> Self {
    Self::Union(DynamicUnionDataflow {
      d1: Box::new(self),
      d2: Box::new(d2),
    })
  }

  pub fn difference(self, d2: Self, ctx: &'a Prov) -> Self {
    Self::Difference(DynamicDifferenceDataflow {
      d1: Box::new(self),
      d2: Box::new(d2),
      ctx,
    })
  }

  pub fn antijoin(self, d2: Self, ctx: &'a Prov) -> Self {
    Self::Antijoin(DynamicAntijoinDataflow {
      d1: Box::new(self),
      d2: Box::new(d2),
      ctx,
    })
  }

  pub fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    match self {
      Self::StableUnit(i) => i.iter_stable(),
      Self::RecentUnit(i) => i.iter_stable(),
      Self::Vec(_) => DynamicBatches::Empty,
      Self::DynamicStableCollection(dc) => dc.iter_stable(),
      Self::DynamicRecentCollection(dc) => dc.iter_stable(),
      Self::DynamicRelation(dr) => dr.iter_stable(),
      Self::OverwriteOne(d) => d.iter_stable(),
      Self::Project(p) => p.iter_stable(),
      Self::Filter(f) => f.iter_stable(),
      Self::Find(f) => f.iter_stable(),
      Self::Intersect(i) => i.iter_stable(),
      Self::Join(j) => j.iter_stable(),
      Self::Product(p) => p.iter_stable(),
      Self::Union(u) => u.iter_stable(),
      Self::Difference(d) => d.iter_stable(),
      Self::Antijoin(a) => a.iter_stable(),
      Self::Aggregate(a) => a.iter_stable(),
    }
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    match self {
      Self::StableUnit(i) => i.iter_recent(),
      Self::RecentUnit(i) => i.iter_recent(),
      Self::Vec(v) => DynamicBatches::single(DynamicBatch::vec(v)),
      Self::DynamicStableCollection(dc) => dc.iter_recent(),
      Self::DynamicRecentCollection(dc) => dc.iter_recent(),
      Self::DynamicRelation(dr) => dr.iter_recent(),
      Self::OverwriteOne(d) => d.iter_recent(),
      Self::Project(p) => p.iter_recent(),
      Self::Filter(f) => f.iter_recent(),
      Self::Find(f) => f.iter_recent(),
      Self::Intersect(i) => i.iter_recent(),
      Self::Join(j) => j.iter_recent(),
      Self::Product(p) => p.iter_recent(),
      Self::Union(u) => u.iter_recent(),
      Self::Difference(d) => d.iter_recent(),
      Self::Antijoin(a) => a.iter_recent(),
      Self::Aggregate(a) => a.iter_recent(),
    }
  }
}

impl<'a, Prov: Provenance> From<&'a DynamicRelation<Prov>> for DynamicDataflow<'a, Prov> {
  fn from(r: &'a DynamicRelation<Prov>) -> Self {
    Self::dynamic_relation(r)
  }
}
