use crate::common::expr::*;
use crate::common::foreign_predicate::*;
use crate::common::tuple::*;
use crate::common::value::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

pub struct ForeignPredicateJoinDataflow<'a, Prov: Provenance> {
  pub left: DynamicDataflow<'a, Prov>,

  /// The foreign predicate
  pub foreign_predicate: String,

  /// The already bounded constants (in order to make this Dataflow free)
  pub args: Vec<Expr>,

  /// Provenance context
  pub ctx: &'a Prov,

  /// Runtime environment
  pub runtime: &'a RuntimeEnvironment
}

impl<'a, Prov: Provenance> Clone for ForeignPredicateJoinDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      left: self.left.clone(),
      foreign_predicate: self.foreign_predicate.clone(),
      args: self.args.clone(),
      ctx: self.ctx,
      runtime: self.runtime,
    }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for ForeignPredicateJoinDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::new(ForeignPredicateJoinBatches {
      batches: self.left.iter_stable(),
      foreign_predicate: self.runtime
        .predicate_registry
        .get(&self.foreign_predicate)
        .expect("Foreign predicate not found")
        .clone(),
      args: self.args.clone(),
      env: self.runtime,
      ctx: self.ctx,
    })
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::new(ForeignPredicateJoinBatches {
      batches: self.left.iter_recent(),
      foreign_predicate: self.runtime
        .predicate_registry
        .get(&self.foreign_predicate)
        .expect("Foreign predicate not found")
        .clone(),
      args: self.args.clone(),
      env: self.runtime,
      ctx: self.ctx,
    })
  }
}

#[derive(Clone)]
pub struct ForeignPredicateJoinBatches<'a, Prov: Provenance> {
  pub batches: DynamicBatches<'a, Prov>,
  pub foreign_predicate: DynamicForeignPredicate,
  pub args: Vec<Expr>,
  pub env: &'a RuntimeEnvironment,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Batches<'a, Prov> for ForeignPredicateJoinBatches<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    // First, try to get a batch from the set of batches
    self.batches.next_batch().map(|mut batch| {
      // Then, try to get the first element inside of this batch;
      // if there is an element, we need to evaluate the foreign predicate and produce a current output batch
      let first_output_batch = batch
        .next_elem()
        .map(|elem| eval_foreign_predicate(elem, &self.foreign_predicate, &self.args, self.env, self.ctx));

      // Generate a new batch
      DynamicBatch::new(ForeignPredicateJoinBatch {
        batch: batch,
        foreign_predicate: self.foreign_predicate.clone(),
        args: self.args.clone(),
        current_output_batch: first_output_batch,
        env: self.env,
        ctx: self.ctx,
      })
    })
  }
}

#[derive(Clone)]
pub struct ForeignPredicateJoinBatch<'a, Prov: Provenance> {
  pub batch: DynamicBatch<'a, Prov>,
  pub foreign_predicate: DynamicForeignPredicate,
  pub args: Vec<Expr>,
  pub current_output_batch: Option<(DynamicElement<Prov>, std::vec::IntoIter<DynamicElement<Prov>>)>,
  pub env: &'a RuntimeEnvironment,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for ForeignPredicateJoinBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    while let Some((left_elem, current_output_batch)) = &mut self.current_output_batch {
      if let Some(right_elem) = current_output_batch.next() {
        let tuple = (left_elem.tuple.clone(), right_elem.tuple);
        let new_tag = self.ctx.mult(&left_elem.tag, &right_elem.tag);
        return Some(DynamicElement::new(tuple, new_tag));
      } else {
        self.current_output_batch = self
          .batch
          .next_elem()
          .map(|elem| eval_foreign_predicate(elem, &self.foreign_predicate, &self.args, self.env, self.ctx));
      }
    }
    None
  }
}

/// Evaluate the foreign predicate on the given element
fn eval_foreign_predicate<Prov: Provenance>(
  elem: DynamicElement<Prov>,
  fp: &DynamicForeignPredicate,
  args: &Vec<Expr>,
  env: &RuntimeEnvironment,
  ctx: &Prov,
) -> (DynamicElement<Prov>, std::vec::IntoIter<DynamicElement<Prov>>) {
  // First get the arguments to pass to the foreign predicate
  let args_to_fp: Vec<Value> = args
    .iter()
    .map(|arg| match arg {
      Expr::Access(a) => elem.tuple[a].as_value(),
      Expr::Constant(c) => c.clone(),
      _ => panic!("Foreign predicate join only supports constant and access arguments"),
    })
    .collect();

  // Then evaluate the foreign predicate on these arguments
  let outputs: Vec<_> = fp
    .evaluate_with_env(env, &args_to_fp)
    .into_iter()
    .filter_map(|(tag, values)| {
      // Make sure to tag the output elements
      let input_tag = Prov::InputTag::from_dynamic_input_tag(&tag);
      let new_tag = ctx.tagging_optional_fn(input_tag);

      // Generate a tuple from the values produced by the foreign predicate
      let tuple = Tuple::from(values);
      let internal_tuple = env.internalize_tuple(&tuple)?;

      // Generate the output element
      Some(DynamicElement::new(internal_tuple, new_tag))
    })
    .collect();

  // Return the input element and output elements pair
  (elem, outputs.into_iter())
}
