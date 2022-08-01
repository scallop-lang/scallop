use std::marker::PhantomData;

use rand::prelude::*;
use rand::rngs::StdRng;

use super::*;
use crate::utils::PointerFamily;

#[derive(Clone)]
pub struct Proofs<T: Clone, P: PointerFamily> {
  pub formula: DNFFormula,
  pub phantom: PhantomData<(T, P)>,
}

impl<T: Clone, P: PointerFamily> PartialEq for Proofs<T, P> {
  fn eq(&self, other: &Self) -> bool {
    self.formula.eq(&other.formula)
  }
}

impl<T: Clone, P: PointerFamily> std::fmt::Debug for Proofs<T, P> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.formula.fmt(f)
  }
}

impl<T: Clone, P: PointerFamily> std::fmt::Display for Proofs<T, P> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.formula.fmt(f)
  }
}

impl<T: Clone, P: PointerFamily> From<DNFFormula> for Proofs<T, P> {
  fn from(formula: DNFFormula) -> Self {
    Self {
      formula,
      phantom: PhantomData,
    }
  }
}

impl<T: Clone + 'static, P: PointerFamily> Tag for Proofs<T, P> {
  type Context = DiffSampleKProofsContext<T, P>;
}

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

impl<T: Clone + 'static, P: PointerFamily> ProvenanceContext for DiffSampleKProofsContext<T, P> {
  type Tag = Proofs<T, P>;

  type InputTag = InputDiffProb<T>;

  type OutputTag = OutputDiffProb<T>;

  fn name() -> &'static str {
    "diff-sample-k-proofs"
  }

  fn tagging_fn(&mut self, input_tag: Self::InputTag) -> Self::Tag {
    let InputDiffProb(p, t) = input_tag;
    let id = self.diff_probs.len();
    P::get_mut(&mut self.diff_probs).push((p, t));
    DNFFormula::singleton(id).into()
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
        DNFFormula::singleton(id).into()
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
    let wmc_result = t.formula.wmc(&s, &v);
    let prob = wmc_result.real;
    let deriv = wmc_result
      .deriv
      .iter()
      .map(|(id, weight)| (id, *weight, self.diff_probs[id].1.clone()))
      .collect::<Vec<_>>();
    OutputDiffProb(prob, deriv)
  }

  fn discard(&self, t: &Self::Tag) -> bool {
    t.formula.is_empty()
  }

  fn zero(&self) -> Self::Tag {
    self.base_zero().into()
  }

  fn one(&self) -> Self::Tag {
    self.base_one().into()
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    P::get_cell_mut(&self.sampler, |sampler| {
      let tag = t1.formula.or(&t2.formula);
      let sampled_clauses = self.sample_k_clauses(tag.clauses, self.k, sampler);
      DNFFormula {
        clauses: sampled_clauses,
      }
      .into()
    })
  }

  fn add_with_proceeding(&self, stable_tag: &Self::Tag, recent_tag: &Self::Tag) -> (Self::Tag, Proceeding) {
    let new_tag = self.add(stable_tag, recent_tag);
    let proceeding = if &new_tag == recent_tag || &new_tag == stable_tag {
      Proceeding::Stable
    } else {
      Proceeding::Recent
    };
    (new_tag, proceeding)
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    P::get_cell_mut(&self.sampler, |sampler| {
      let mut tag = t1.formula.or(&t2.formula);
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

  fn minus(&self, _: &Self::Tag, _: &Self::Tag) -> Option<Self::Tag> {
    panic!("Not implemented")
  }
}
