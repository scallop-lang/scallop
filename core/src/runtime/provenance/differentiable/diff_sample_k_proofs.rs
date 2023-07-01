use rand::prelude::*;
use rand::rngs::StdRng;

use crate::common::tensors::*;
use crate::utils::PointerFamily;

use super::*;

pub struct DiffSampleKProofsProvenance<T: FromTensor, P: PointerFamily> {
  pub k: usize,
  pub sampler: P::Cell<StdRng>,
  pub storage: DiffProbStorage<T, P>,
  pub disjunctions: P::Cell<Disjunctions>,
}

impl<T: FromTensor, P: PointerFamily> Clone for DiffSampleKProofsProvenance<T, P> {
  fn clone(&self) -> Self {
    Self {
      k: self.k,
      sampler: P::clone_cell(&self.sampler),
      storage: self.storage.clone_internal(),
      disjunctions: P::clone_cell(&self.disjunctions),
    }
  }
}

impl<T: FromTensor, P: PointerFamily> DiffSampleKProofsProvenance<T, P> {
  pub fn new(k: usize) -> Self {
    Self::new_with_seed(k, 12345678)
  }

  pub fn new_with_seed(k: usize, seed: u64) -> Self {
    Self {
      k,
      sampler: P::new_cell(StdRng::seed_from_u64(seed)),
      storage: DiffProbStorage::new(),
      disjunctions: P::new_cell(Disjunctions::new()),
    }
  }

  pub fn input_tags(&self) -> Vec<T> {
    self.storage.input_tags()
  }

  pub fn set_k(&mut self, new_k: usize) {
    self.k = new_k;
  }
}

impl<T: FromTensor, P: PointerFamily> DNFContextTrait for DiffSampleKProofsProvenance<T, P> {
  fn fact_probability(&self, id: &usize) -> f64 {
    self.storage.fact_probability(id)
  }

  fn has_disjunction_conflict(&self, pos_facts: &std::collections::BTreeSet<usize>) -> bool {
    P::get_cell(&self.disjunctions, |d| d.has_conflict(pos_facts))
  }
}

impl<T: FromTensor, P: PointerFamily> Provenance for DiffSampleKProofsProvenance<T, P> {
  type Tag = DNFFormula;

  type InputTag = InputExclusiveDiffProb<T>;

  type OutputTag = OutputDiffProb;

  fn name() -> &'static str {
    "diff-sample-k-proofs"
  }

  fn tagging_fn(&self, input_tag: Self::InputTag) -> Self::Tag {
    let InputExclusiveDiffProb {
      prob,
      external_tag,
      exclusion,
    } = input_tag;

    // First store the probability and generate the id
    let fact_id = self.storage.add_prob(prob, external_tag);

    // Store the mutual exclusivity
    if let Some(disjunction_id) = exclusion {
      P::get_cell_mut(&self.disjunctions, |d| d.add_disjunction(disjunction_id, fact_id));
    }

    // Finally return the formula
    DNFFormula::singleton(fact_id)
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    // Get the number of variables that requires grad
    let num_var_requires_grad = self.storage.num_input_tags();
    let s = DualNumberSemiring::new(num_var_requires_grad);
    let v = |i: &usize| {
      let (real, external_tag) = self.storage.get_diff_prob(i);

      // Check if this variable `i` requires grad or not
      if external_tag.is_some() {
        s.singleton(real.clone(), i.clone())
      } else {
        s.constant(real.clone())
      }
    };
    let wmc_result = t.wmc(&s, &v);
    let prob = wmc_result.real;
    let deriv = wmc_result
      .deriv
      .iter()
      .map(|(id, weight)| (id, *weight))
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

  fn weight(&self, t: &Self::Tag) -> f64 {
    let v = |i: &usize| self.storage.get_prob(i);
    t.wmc(&RealSemiring::new(), &v)
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
