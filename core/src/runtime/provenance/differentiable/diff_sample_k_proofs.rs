use rand::prelude::*;
use rand::rngs::StdRng;

use super::*;
use crate::utils::PointerFamily;

pub struct DiffSampleKProofsContext<T: Clone, P: PointerFamily> {
  pub k: usize,
  pub sampler: P::Cell<StdRng>,
  pub diff_probs: P::Pointer<Vec<(f64, T)>>,
  pub disjunctions: Disjunctions,
}

impl<T: Clone, P: PointerFamily> Clone for DiffSampleKProofsContext<T, P> {
  fn clone(&self) -> Self {
    Self {
      k: self.k,
      sampler: P::clone_cell(&self.sampler),
      diff_probs: P::new((&*self.diff_probs).clone()),
      disjunctions: self.disjunctions.clone(),
    }
  }
}

impl<T: Clone, P: PointerFamily> DiffSampleKProofsContext<T, P> {
  pub fn new(k: usize) -> Self {
    Self::new_with_seed(k, 12345678)
  }

  pub fn new_with_seed(k: usize, seed: u64) -> Self {
    Self {
      k,
      sampler: P::new_cell(StdRng::seed_from_u64(seed)),
      diff_probs: P::new(Vec::new()),
      disjunctions: Disjunctions::new(),
    }
  }

  pub fn input_tags(&self) -> Vec<T> {
    self.diff_probs.iter().map(|(_, t)| t.clone()).collect()
  }

  pub fn set_k(&mut self, new_k: usize) {
    self.k = new_k;
  }
}

impl<T: Clone, P: PointerFamily> DNFContextTrait for DiffSampleKProofsContext<T, P> {
  fn fact_probability(&self, id: &usize) -> f64 {
    self.diff_probs[id.clone()].0
  }

  fn has_disjunction_conflict(&self, pos_facts: &std::collections::BTreeSet<usize>) -> bool {
    self.disjunctions.has_conflict(pos_facts)
  }
}

impl<T: Clone + 'static, P: PointerFamily> Provenance for DiffSampleKProofsContext<T, P> {
  type Tag = DNFFormula;

  type InputTag = InputDiffProb<T>;

  type OutputTag = OutputDiffProb<T>;

  fn name() -> &'static str {
    "diff-sample-k-proofs"
  }

  fn tagging_fn(&mut self, input_tag: Self::InputTag) -> Self::Tag {
    let InputDiffProb(p, t) = input_tag;
    let id = self.diff_probs.len();
    P::get_mut(&mut self.diff_probs).push((p, t));
    DNFFormula::singleton(id)
  }

  fn tagging_disjunction_fn(&mut self, tags: Vec<Self::InputTag>) -> Vec<Self::Tag> {
    let mut ids = vec![];

    // Add base disjunctions
    let tags = tags
      .into_iter()
      .map(|tag| {
        let InputDiffProb(p, t) = tag;
        let id = self.diff_probs.len();
        P::get_mut(&mut self.diff_probs).push((p, t));
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
    let s = DualNumberSemiring::new(self.diff_probs.len());
    let v = |i: &usize| {
      let (real, _) = &self.diff_probs[i.clone()];
      s.singleton(real.clone(), i.clone())
    };
    let wmc_result = t.wmc(&s, &v);
    let prob = wmc_result.real;
    let deriv = wmc_result
      .deriv
      .iter()
      .map(|(id, weight)| (id, *weight, self.diff_probs[id].1.clone()))
      .collect::<Vec<_>>();
    OutputDiffProb(prob, deriv)
  }

  fn discard(&self, t: &Self::Tag) -> bool {
    t.is_empty()
  }

  fn zero(&self) -> Self::Tag {
    self.base_zero().into()
  }

  fn one(&self) -> Self::Tag {
    self.base_one().into()
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    P::get_cell_mut(&self.sampler, |sampler| {
      let tag = t1.or(t2);
      let sampled_clauses = self.sample_k_clauses(tag.clauses, self.k, sampler);
      DNFFormula {
        clauses: sampled_clauses,
      }
      .into()
    })
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    t_old == t_new
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    P::get_cell_mut(&self.sampler, |sampler| {
      let mut tag = t1.and(t2);
      self.retain_no_conflict(&mut tag.clauses);
      let sampled_clauses = self.sample_k_clauses(tag.clauses, self.k, sampler);
      DNFFormula {
        clauses: sampled_clauses,
      }
      .into()
    })
  }

  fn negate(&self, _: &Self::Tag) -> Option<Self::Tag> {
    panic!("Not implemented")
  }
}
