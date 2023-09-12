use crate::common::expr::*;
use crate::common::tuple::*;
use crate::common::tuple_type::*;
use crate::common::value::*;

use super::*;

#[allow(unused)]
pub trait Dataflow<'a, Prov: Provenance>: 'a + dyn_clone::DynClone {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::empty()
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::empty()
  }
}

pub struct DynamicDataflow<'a, Prov: Provenance>(Box<dyn Dataflow<'a, Prov>>);

impl<'a, Prov: Provenance> Clone for DynamicDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self(dyn_clone::clone_box(&*self.0))
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicDataflow<'a, Prov> {
  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    self.0.iter_recent()
  }

  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    self.0.iter_stable()
  }
}

impl<'a, Prov: Provenance> DynamicDataflow<'a, Prov> {
  pub fn new<D: Dataflow<'a, Prov>>(d: D) -> Self {
    Self(Box::new(d))
  }

  pub fn vec(vec: &'a DynamicElements<Prov>) -> Self {
    Self::new(RefElementsDataflow(vec))
  }

  pub fn untagged_vec(ctx: &'a Prov, vec: Vec<Tuple>) -> Self {
    Self::new(DynamicUntaggedVec::new(ctx, vec))
  }

  pub fn recent_unit(ctx: &'a Prov, tuple_type: TupleType) -> Self {
    Self::new(DynamicRecentUnitDataflow::new(ctx, tuple_type))
  }

  pub fn stable_unit(ctx: &'a Prov, tuple_type: TupleType) -> Self {
    Self::new(DynamicStableUnitDataflow::new(ctx, tuple_type))
  }

  pub fn dynamic_collection(col: &'a DynamicCollection<Prov>, recent: bool) -> Self {
    if recent {
      Self::dynamic_recent_collection(col)
    } else {
      Self::dynamic_stable_collection(col)
    }
  }

  pub fn dynamic_stable_collection(col: &'a DynamicCollection<Prov>) -> Self {
    Self::new(DynamicStableCollectionDataflow(col))
  }

  pub fn dynamic_recent_collection(col: &'a DynamicCollection<Prov>) -> Self {
    Self::new(DynamicRecentCollectionDataflow(col))
  }

  pub fn dynamic_relation(rela: &'a DynamicRelation<Prov>) -> Self {
    Self::new(DynamicRelationDataflow(rela))
  }

  pub fn overwrite_one(self, ctx: &'a Prov) -> Self {
    Self::new(DynamicOverwriteOneDataflow { source: self, ctx })
  }

  pub fn project(self, expression: Expr, runtime: &'a RuntimeEnvironment) -> Self {
    Self::new(DynamicProjectDataflow { source: self, expression, runtime })
  }

  pub fn filter(self, filter: Expr, runtime: &'a RuntimeEnvironment) -> Self {
    Self::new(DynamicFilterDataflow { source: self, filter, runtime })
  }

  pub fn find(self, key: Tuple) -> Self {
    Self::new(DynamicFindDataflow { source: self, key })
  }

  pub fn intersect(self, d2: Self, ctx: &'a Prov) -> Self {
    Self::new(DynamicIntersectDataflow { d1: self, d2, ctx })
  }

  pub fn join(self, d2: Self, ctx: &'a Prov) -> Self {
    Self::new(DynamicJoinDataflow { d1: self, d2, ctx })
  }

  pub fn product(self, d2: Self, ctx: &'a Prov) -> Self {
    Self::new(DynamicProductDataflow { d1: self, d2, ctx })
  }

  pub fn union(self, d2: Self) -> Self {
    Self::new(DynamicUnionDataflow { d1: self, d2 })
  }

  pub fn difference(self, d2: Self, ctx: &'a Prov) -> Self {
    Self::new(DynamicDifferenceDataflow { d1: self, d2, ctx })
  }

  pub fn antijoin(self, d2: Self, ctx: &'a Prov) -> Self {
    Self::new(DynamicAntijoinDataflow { d1: self, d2, ctx })
  }

  pub fn foreign_predicate_ground(pred: String, bounded: Vec<Value>, first_iter: bool, ctx: &'a Prov, runtime: &'a RuntimeEnvironment) -> Self {
    Self::new(ForeignPredicateGroundDataflow {
      foreign_predicate: pred,
      bounded_constants: bounded,
      first_iteration: first_iter,
      ctx,
      runtime,
    })
  }

  pub fn foreign_predicate_constraint(self, pred: String, args: Vec<Expr>, ctx: &'a Prov, runtime: &'a RuntimeEnvironment) -> Self {
    Self::new(ForeignPredicateConstraintDataflow {
      dataflow: self,
      foreign_predicate: pred,
      args,
      ctx,
      runtime,
    })
  }

  pub fn foreign_predicate_join(self, pred: String, args: Vec<Expr>, ctx: &'a Prov, runtime: &'a RuntimeEnvironment) -> Self {
    Self::new(ForeignPredicateJoinDataflow {
      left: self,
      foreign_predicate: pred,
      args,
      ctx,
      runtime,
    })
  }

  pub fn dynamic_exclusion(self, other: Self, ctx: &'a Prov, runtime: &'a RuntimeEnvironment) -> Self {
    Self::new(DynamicExclusionDataflow::new(self, other, ctx, runtime))
  }
}

#[derive(Clone)]
pub struct RefElementsDataflow<'a, Prov: Provenance>(&'a DynamicElements<Prov>);

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for RefElementsDataflow<'a, Prov> {
  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::single(RefElementsBatch::new(self.0))
  }
}
