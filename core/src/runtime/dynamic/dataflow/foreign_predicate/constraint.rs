use crate::common::expr::*;
use crate::common::foreign_predicate::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct ForeignPredicateConstraintDataflow<'a, Prov: Provenance> {
  /// Sub-dataflow
  pub dataflow: DynamicDataflow<'a, Prov>,

  /// The foreign predicate
  pub foreign_predicate: String,

  /// The arguments to the foreign predicate
  pub args: Vec<Expr>,

  /// Provenance context
  pub ctx: &'a Prov,

  /// Runtime environment
  pub runtime: &'a RuntimeEnvironment
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for ForeignPredicateConstraintDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    let fp = self.runtime
      .predicate_registry
      .get(&self.foreign_predicate)
      .expect("Foreign predicate not found");
    DynamicBatches::new(ForeignPredicateConstraintBatches {
      batches: self.dataflow.iter_stable(),
      foreign_predicate: fp.clone(),
      args: self.args.clone(),
      env: self.runtime,
      ctx: self.ctx,
    })
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    let fp = self.runtime
      .predicate_registry
      .get(&self.foreign_predicate)
      .expect("Foreign predicate not found");
    DynamicBatches::new(ForeignPredicateConstraintBatches {
      batches: self.dataflow.iter_recent(),
      foreign_predicate: fp.clone(),
      args: self.args.clone(),
      env: self.runtime,
      ctx: self.ctx,
    })
  }
}

#[derive(Clone)]
pub struct ForeignPredicateConstraintBatches<'a, Prov: Provenance> {
  pub batches: DynamicBatches<'a, Prov>,
  pub foreign_predicate: DynamicForeignPredicate,
  pub args: Vec<Expr>,
  pub env: &'a RuntimeEnvironment,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Batches<'a, Prov> for ForeignPredicateConstraintBatches<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    self.batches.next_batch().map(|batch| {
      DynamicBatch::new(ForeignPredicateConstraintBatch {
        batch: batch,
        foreign_predicate: self.foreign_predicate.clone(),
        args: self.args.clone(),
        env: self.env,
        ctx: self.ctx,
      })
    })
  }
}

#[derive(Clone)]
pub struct ForeignPredicateConstraintBatch<'a, Prov: Provenance> {
  pub batch: DynamicBatch<'a, Prov>,
  pub foreign_predicate: DynamicForeignPredicate,
  pub args: Vec<Expr>,
  pub env: &'a RuntimeEnvironment,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for ForeignPredicateConstraintBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    while let Some(elem) = self.batch.next_elem() {
      let Tagged { tuple, tag } = elem;

      // Try evaluate the arguments; if failed, continue to the next element in the batch
      let values = self
        .args
        .iter()
        .map(|arg| match arg {
          Expr::Access(a) => tuple[a].as_value(),
          Expr::Constant(c) => c.clone(),
          _ => panic!("Invalid argument to bounded foreign predicate"),
        })
        .collect::<Vec<_>>();

      // Evaluate the foreign predicate to produce a list of output tags
      // Note that there will be at most one output tag since the foreign predicate is bounded
      let result = self.foreign_predicate.evaluate_with_env(self.env, &values);

      // Check if the foreign predicate returned a tag
      if !result.is_empty() {
        assert_eq!(
          result.len(),
          1,
          "Bounded foreign predicate should return at most one element per evaluation"
        );
        let input_tag = Prov::InputTag::from_dynamic_input_tag(&result[0].0);
        let new_tag = self.ctx.tagging_optional_fn(input_tag);
        let combined_tag = self.ctx.mult(&tag, &new_tag);
        return Some(DynamicElement::new(tuple, combined_tag));
      }
    }
    None
  }
}
