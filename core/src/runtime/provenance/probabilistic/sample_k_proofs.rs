use std::cell::RefCell;
use std::collections::*;
use std::rc::Rc;

use rand::prelude::*;
use rand::rngs::StdRng;

use super::*;

pub struct SampleKProofsProvenance {
  pub k: usize,
  pub sampler: Rc<RefCell<StdRng>>,
  pub probs: Rc<Vec<f64>>,
  pub disjunctions: Disjunctions,
}

impl Clone for SampleKProofsProvenance {
  fn clone(&self) -> Self {
    Self {
      k: self.k,
      sampler: self.sampler.clone(),
      probs: Rc::new((&*self.probs).clone()),
      disjunctions: self.disjunctions.clone(),
    }
  }
}

impl SampleKProofsProvenance {
  pub fn new(k: usize) -> Self {
    Self::new_with_seed(k, 12345678)
  }

  pub fn new_with_seed(k: usize, seed: u64) -> Self {
    Self {
      k,
      sampler: Rc::new(RefCell::new(StdRng::seed_from_u64(seed))),
      probs: Rc::new(Vec::new()),
      disjunctions: Disjunctions::new(),
    }
  }

  pub fn set_k(&mut self, new_k: usize) {
    self.k = new_k;
  }
}

impl DNFContextTrait for SampleKProofsProvenance {
  fn fact_probability(&self, id: &usize) -> f64 {
    self.probs[*id]
  }

  fn has_disjunction_conflict(&self, pos_facts: &BTreeSet<usize>) -> bool {
    self.disjunctions.has_conflict(pos_facts)
  }
}

impl Provenance for SampleKProofsProvenance {
  type Tag = DNFFormula;

  type InputTag = InputExclusiveProb;

  type OutputTag = f64;

  fn name() -> &'static str {
    "sample-k-proofs"
  }

  fn tagging_fn(&mut self, input_tag: Self::InputTag) -> Self::Tag {
    // First generate id and push the probability into the list
    let fact_id = self.probs.len();
    Rc::get_mut(&mut self.probs).unwrap().push(input_tag.prob);

    // Add exlusion if needed
    if let Some(disj_id) = input_tag.exclusion {
      self.disjunctions.add_disjunction(disj_id, fact_id);
    }

    // Lastly return a tag
    Self::Tag::singleton(fact_id)
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    let s = RealSemiring;
    let v = |i: &usize| -> f64 { self.probs[*i] };
    t.wmc(&s, &v)
  }

  fn discard(&self, t: &Self::Tag) -> bool {
    t.is_empty()
  }

  fn zero(&self) -> Self::Tag {
    self.base_zero()
  }

  fn one(&self) -> Self::Tag {
    self.base_one()
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    let tag = t1.or(t2);
    let sampled_clauses = self.sample_k_clauses(tag.clauses, self.k, &mut self.sampler.borrow_mut());
    DNFFormula {
      clauses: sampled_clauses,
    }
  }

  fn saturated(&self, _: &Self::Tag, _: &Self::Tag) -> bool {
    true
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    let mut tag = t1.or(t2);
    self.retain_no_conflict(&mut tag.clauses);
    let sampled_clauses = self.sample_k_clauses(tag.clauses, self.k, &mut self.sampler.borrow_mut());
    DNFFormula {
      clauses: sampled_clauses,
    }
  }

  fn negate(&self, _: &Self::Tag) -> Option<Self::Tag> {
    panic!("Not implemented")
  }

  fn minus(&self, _: &Self::Tag, _: &Self::Tag) -> Option<Self::Tag> {
    panic!("Not implemented")
  }

  fn weight(&self, t: &Self::Tag) -> f64 {
    let s = RealSemiring;
    let v = |i: &usize| -> f64 { self.probs[*i] };
    t.wmc(&s, &v)
  }
}
