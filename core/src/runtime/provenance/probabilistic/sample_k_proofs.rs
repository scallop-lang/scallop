use std::cell::RefCell;
use std::collections::*;
use std::rc::Rc;

use rand::prelude::*;
use rand::rngs::StdRng;

use super::*;

pub struct SampleKProofsContext {
  pub k: usize,
  pub sampler: Rc<RefCell<StdRng>>,
  pub probs: Rc<Vec<f64>>,
  pub disjunctions: Disjunctions,
}

impl Clone for SampleKProofsContext {
  fn clone(&self) -> Self {
    Self {
      k: self.k,
      sampler: self.sampler.clone(),
      probs: Rc::new((&*self.probs).clone()),
      disjunctions: self.disjunctions.clone(),
    }
  }
}

impl SampleKProofsContext {
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

impl DNFContextTrait for SampleKProofsContext {
  fn fact_probability(&self, id: &usize) -> f64 {
    self.probs[*id]
  }

  fn has_disjunction_conflict(&self, pos_facts: &BTreeSet<usize>) -> bool {
    self.disjunctions.has_conflict(pos_facts)
  }
}

impl Provenance for SampleKProofsContext {
  type Tag = DNFFormula;

  type InputTag = f64;

  type OutputTag = f64;

  fn name() -> &'static str {
    "sample-k-proofs"
  }

  fn tagging_fn(&mut self, prob: Self::InputTag) -> Self::Tag {
    let id = self.probs.len();
    Rc::get_mut(&mut self.probs).unwrap().push(prob);
    DNFFormula::singleton(id)
  }

  fn tagging_disjunction_fn(&mut self, tags: Vec<Self::InputTag>) -> Vec<Self::Tag> {
    let mut ids = vec![];

    // Add base disjunctions
    let tags = tags
      .into_iter()
      .map(|tag| {
        let id = self.probs.len();
        Rc::get_mut(&mut self.probs).unwrap().push(tag);
        ids.push(id);
        DNFFormula::singleton(id)
      })
      .collect::<Vec<_>>();

    // Add disjunction
    self.disjunctions.add_disjunction(ids.clone().into_iter());

    // Return tags
    tags
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

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    t_old == t_new
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
}
