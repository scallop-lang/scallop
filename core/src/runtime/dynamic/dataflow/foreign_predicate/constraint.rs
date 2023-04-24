use crate::common::expr::*;
use crate::common::foreign_predicate::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct ForeignPredicateConstraintDataflow<'a, Prov: Provenance> {
  /// Sub-dataflow
  pub dataflow: Box<DynamicDataflow<'a, Prov>>,

  /// The foreign predicate
  pub foreign_predicate: String,

  /// The arguments to the foreign predicate
  pub args: Vec<Expr>,

  /// Provenance context
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> ForeignPredicateConstraintDataflow<'a, Prov> {
  pub fn iter_stable(&self, runtime: &'a RuntimeEnvironment) -> DynamicBatches<'a, Prov> {
    let fp = runtime.predicate_registry.get(&self.foreign_predicate).expect("Foreign predicate not found");
    DynamicBatches::ForeignPredicateConstraint(ForeignPredicateConstraintBatches {
      batches: Box::new(self.dataflow.iter_stable(runtime)),
      foreign_predicate: fp.clone(),
      args: self.args.clone(),
      ctx: self.ctx,
    })
  }

  pub fn iter_recent(&self, runtime: &'a RuntimeEnvironment) -> DynamicBatches<'a, Prov> {
    let fp = runtime.predicate_registry.get(&self.foreign_predicate).expect("Foreign predicate not found");
    DynamicBatches::ForeignPredicateConstraint(ForeignPredicateConstraintBatches {
      batches: Box::new(self.dataflow.iter_recent(runtime)),
      foreign_predicate: fp.clone(),
      args: self.args.clone(),
      ctx: self.ctx,
    })
  }
}

#[derive(Clone)]
pub struct ForeignPredicateConstraintBatches<'a, Prov: Provenance> {
  pub batches: Box<DynamicBatches<'a, Prov>>,
  pub foreign_predicate: DynamicForeignPredicate,
  pub args: Vec<Expr>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Iterator for ForeignPredicateConstraintBatches<'a, Prov> {
  type Item = DynamicBatch<'a, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    self.batches.next().map(|batch| {
      DynamicBatch::ForeignPredicateConstraint(ForeignPredicateConstraintBatch {
        batch: Box::new(batch),
        foreign_predicate: self.foreign_predicate.clone(),
        args: self.args.clone(),
        ctx: self.ctx,
      })
    })
  }
}

#[derive(Clone)]
pub struct ForeignPredicateConstraintBatch<'a, Prov: Provenance> {
  pub batch: Box<DynamicBatch<'a, Prov>>,
  pub foreign_predicate: DynamicForeignPredicate,
  pub args: Vec<Expr>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Iterator for ForeignPredicateConstraintBatch<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    while let Some(elem) = self.batch.next() {
      let Tagged { tuple, tag } = elem;

      // Try evaluate the arguments; if failed, continue to the next element in the batch
      let values = self.args.iter().map(|arg| {
        match arg {
          Expr::Access(a) => tuple[a].as_value(),
          Expr::Constant(c) => c.clone(),
          _ => panic!("Invalid argument to bounded foreign predicate")
        }
      }).collect::<Vec<_>>();

      // Evaluate the foreign predicate to produce a list of output tags
      // Note that there will be at most one output tag since the foreign predicate is bounded
      let result = self.foreign_predicate.evaluate(&values);

      // Check if the foreign predicate returned a tag
      if !result.is_empty() {
        assert_eq!(result.len(), 1, "Bounded foreign predicate should return at most one element per evaluation");
        let input_tag = Prov::InputTag::from_dynamic_input_tag(&result[0].0);
        let new_tag = self.ctx.tagging_optional_fn(input_tag);
        let combined_tag = self.ctx.mult(&tag, &new_tag);
        return Some(DynamicElement::new(tuple, combined_tag));
      }
    }
    None
  }
}
