use crate::common::foreign_predicate::*;
use crate::common::input_tag::*;
use crate::common::expr::*;
use crate::common::tuple::*;
use crate::common::value::*;
use crate::runtime::provenance::*;
use crate::runtime::env::*;

use super::*;

pub struct ForeignPredicateJoinDataflow<'a, Prov: Provenance> {
  pub left: Box<DynamicDataflow<'a, Prov>>,

  /// The foreign predicate
  pub foreign_predicate: String,

  /// The already bounded constants (in order to make this Dataflow free)
  pub args: Vec<Expr>,

  /// Provenance context
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for ForeignPredicateJoinDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      left: self.left.clone(),
      foreign_predicate: self.foreign_predicate.clone(),
      args: self.args.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> ForeignPredicateJoinDataflow<'a, Prov> {
  pub fn iter_stable(&self, runtime: &'a RuntimeEnvironment) -> DynamicBatches<'a, Prov> {
    DynamicBatches::ForeignPredicateJoin(ForeignPredicateJoinBatches {
      batches: Box::new(self.left.iter_stable(runtime)),
      foreign_predicate: runtime.predicate_registry.get(&self.foreign_predicate).expect("Foreign predicate not found").clone(),
      args: self.args.clone(),
      ctx: self.ctx,
    })
  }

  pub fn iter_recent(&self, runtime: &'a RuntimeEnvironment) -> DynamicBatches<'a, Prov> {
    DynamicBatches::ForeignPredicateJoin(ForeignPredicateJoinBatches {
      batches: Box::new(self.left.iter_recent(runtime)),
      foreign_predicate: runtime.predicate_registry.get(&self.foreign_predicate).expect("Foreign predicate not found").clone(),
      args: self.args.clone(),
      ctx: self.ctx,
    })
  }
}

#[derive(Clone)]
pub struct ForeignPredicateJoinBatches<'a, Prov: Provenance> {
  pub batches: Box<DynamicBatches<'a, Prov>>,
  pub foreign_predicate: DynamicForeignPredicate,
  pub args: Vec<Expr>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Iterator for ForeignPredicateJoinBatches<'a, Prov> {
  type Item = DynamicBatch<'a, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    // First, try to get a batch from the set of batches
    self.batches.next().map(|mut batch| {
      // Then, try to get the first element inside of this batch;
      // if there is an element, we need to evaluate the foreign predicate and produce a current output batch
      let first_output_batch = batch.next().map(|elem| {
        eval_foreign_predicate(elem, &self.foreign_predicate, &self.args, self.ctx)
      });

      // Generate a new batch
      DynamicBatch::ForeignPredicateJoin(ForeignPredicateJoinBatch {
        batch: Box::new(batch),
        foreign_predicate: self.foreign_predicate.clone(),
        args: self.args.clone(),
        current_output_batch: first_output_batch,
        ctx: self.ctx,
      })
    })
  }
}

#[derive(Clone)]
pub struct ForeignPredicateJoinBatch<'a, Prov: Provenance> {
  pub batch: Box<DynamicBatch<'a, Prov>>,
  pub foreign_predicate: DynamicForeignPredicate,
  pub args: Vec<Expr>,
  pub current_output_batch: Option<(DynamicElement<Prov>, std::vec::IntoIter<DynamicElement<Prov>>)>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Iterator for ForeignPredicateJoinBatch<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    while let Some((left_elem, current_output_batch)) = &mut self.current_output_batch {
      if let Some(right_elem) = current_output_batch.next() {
        let tuple = (left_elem.tuple.clone(), right_elem.tuple);
        let new_tag = self.ctx.mult(&left_elem.tag, &right_elem.tag);
        return Some(DynamicElement::new(tuple, new_tag))
      } else {
        self.current_output_batch = self.batch.next().map(|elem| {
          eval_foreign_predicate(elem, &self.foreign_predicate, &self.args, self.ctx)
        });
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
  ctx: &Prov,
) -> (DynamicElement<Prov>, std::vec::IntoIter<DynamicElement<Prov>>) {
  // First get the arguments to pass to the foreign predicate
  let args_to_fp: Vec<Value> = args.iter().map(|arg| {
    match arg {
      Expr::Access(a) => elem.tuple[a].as_value(),
      Expr::Constant(c) => c.clone(),
      _ => panic!("Foreign predicate join only supports constant and access arguments"),
    }
  }).collect();

  // Then evaluate the foreign predicate on these arguments
  let outputs: Vec<_> = fp.evaluate(&args_to_fp).into_iter().map(|(tag, values)| {
    // Make sure to tag the output elements
    let input_tag = Prov::InputTag::from_dynamic_input_tag(&tag);
    let new_tag = ctx.tagging_optional_fn(input_tag);

    // Generate a tuple from the values produced by the foreign predicate
    let tuple = Tuple::from(values);

    // Generate the output element
    DynamicElement::new(tuple, new_tag)
  }).collect();

  // Return the input element and output elements pair
  (elem, outputs.into_iter())
}
