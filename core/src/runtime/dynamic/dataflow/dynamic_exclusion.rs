use std::collections::*;

use crate::common::input_tag::*;
use crate::common::tuple::*;
use crate::runtime::provenance::*;
use crate::utils::*;

use super::*;

type VisitedExclusionMap = <RcFamily as PointerFamily>::RcCell<HashMap<Tuple, usize>>;

#[derive(Clone)]
pub struct DynamicExclusionDataflow<'a, Prov: Provenance> {
  // The left dataflow which generates base tuples
  pub left: Box<DynamicDataflow<'a, Prov>>,

  // The right dataflow which generates tuples for exclusion
  // Current assumption is that the right dataflow is an `UntaggedVec`
  pub right: Box<DynamicDataflow<'a, Prov>>,

  // The provenance context
  pub ctx: &'a Prov,

  // A map of tuples that have been visited to their mutual exclusion IDs
  visited: VisitedExclusionMap,
}

impl<'a, Prov: Provenance> DynamicExclusionDataflow<'a, Prov> {
  pub fn new(left: DynamicDataflow<'a, Prov>, right: DynamicDataflow<'a, Prov>, ctx: &'a Prov) -> Self {
    Self {
      left: Box::new(left),
      right: Box::new(right),
      ctx,
      visited: RcFamily::new_rc_cell(HashMap::new()),
    }
  }

  pub fn iter_recent(&self, env: &'a RuntimeEnvironment) -> DynamicBatches<'a, Prov> {
    let left = self.left.iter_recent(env);
    let right = self.right.iter_recent(env);
    let op = ExclusionOp::new(env, self.ctx, RcFamily::clone_rc_cell(&self.visited));
    DynamicBatches::binary(left, right, BatchBinaryOp::Exclusion(op))
  }

  pub fn iter_stable(&self, env: &'a RuntimeEnvironment) -> DynamicBatches<'a, Prov> {
    let left = self.left.iter_stable(env);
    let right = self.right.iter_recent(env);
    let op = ExclusionOp::new(env, self.ctx, RcFamily::clone_rc_cell(&self.visited));
    DynamicBatches::binary(left, right, BatchBinaryOp::Exclusion(op))
  }
}

#[derive(Clone)]
pub struct ExclusionOp<'a, Prov: Provenance> {
  pub runtime: &'a RuntimeEnvironment,
  pub ctx: &'a Prov,
  pub visited_exclusion_map: VisitedExclusionMap,
}

impl<'a, Prov: Provenance> ExclusionOp<'a, Prov> {
  pub fn new(runtime: &'a RuntimeEnvironment, ctx: &'a Prov, visited_exclusion_map: VisitedExclusionMap) -> Self {
    Self {
      runtime,
      ctx,
      visited_exclusion_map,
    }
  }

  pub fn apply(&self, left: DynamicBatch<'a, Prov>, right: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    DynamicBatch::Exclusion(DynamicExclusionBatch::new(
      self.runtime,
      self.ctx,
      RcFamily::clone_rc_cell(&self.visited_exclusion_map),
      left,
      right,
    ))
  }
}

#[derive(Clone)]
pub struct DynamicExclusionBatch<'a, Prov: Provenance> {
  // Basic information
  pub runtime: &'a RuntimeEnvironment,
  pub ctx: &'a Prov,
  pub visited_exclusion_map: VisitedExclusionMap,

  // Batches
  pub left: Box<DynamicBatch<'a, Prov>>,
  pub left_curr: Option<DynamicElement<Prov>>,
  pub right_source: Box<DynamicBatch<'a, Prov>>,
  pub right_clone: Box<DynamicBatch<'a, Prov>>,
  pub curr_exclusion_id: Option<usize>,
}

impl<'a, Prov: Provenance> DynamicExclusionBatch<'a, Prov> {
  pub fn new(
    runtime: &'a RuntimeEnvironment,
    ctx: &'a Prov,
    visited_exclusion_map: VisitedExclusionMap,
    mut left: DynamicBatch<'a, Prov>,
    right: DynamicBatch<'a, Prov>,
  ) -> Self {
    let right_clone = right.clone();
    let left_curr = left.next();
    Self {
      runtime,
      ctx,
      visited_exclusion_map,
      left: Box::new(left),
      left_curr,
      right_source: Box::new(right),
      right_clone: Box::new(right_clone),
      curr_exclusion_id: None,
    }
  }
}

impl<'a, Prov: Provenance> Iterator for DynamicExclusionBatch<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    loop {
      if let Some(left) = &self.left_curr {
        // First get an exclusion ID
        let exc_id =
          if let Some(id) = RcFamily::get_rc_cell(&self.visited_exclusion_map, |m| m.get(&left.tuple).cloned()) {
            // If the left tuple has been visited, directly pull the exclusion id
            id
          } else if let Some(id) = self.curr_exclusion_id {
            // Or we have already generated a new ID for this tuple
            id
          } else {
            // Otherwise, generate a new ID
            let id = self.runtime.allocate_new_exclusion_id();
            RcFamily::get_rc_cell_mut(&self.visited_exclusion_map, |m| m.insert(left.tuple.clone(), id));
            id
          };

        // Then, iterate through the right
        if let Some(right) = self.right_clone.next() {
          // Create a tuple combining left and right
          let tuple: Tuple = (left.tuple.clone(), right.tuple.clone()).into();
          let me_input_tag = Prov::InputTag::from_dynamic_input_tag(&DynamicInputTag::Exclusive(exc_id));
          let me_tag = self.ctx.tagging_optional_fn(me_input_tag);
          let tag = self.ctx.mult(&left.tag, &me_tag);
          return Some(DynamicElement::new(tuple, tag));
        } else {
          // Move on to the next left element and reset other states
          self.left_curr = self.left.next();
          self.right_clone = self.right_source.clone();
          self.curr_exclusion_id = None;
        }
      } else {
        return None;
      }
    }
  }
}
