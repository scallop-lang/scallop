use super::*;
use crate::utils::*;

pub struct TopKProofsProvenance<P: PointerFamily = RcFamily> {
  pub k: usize,
  pub probs: P::Cell<Vec<f64>>,
  pub disjunctions: P::Cell<Disjunctions>,
  pub wmc_with_disjunctions: bool,
}

impl<P: PointerFamily> Default for TopKProofsProvenance<P> {
  fn default() -> Self {
    Self {
      k: 3,
      probs: P::new_cell(Vec::new()),
      disjunctions: P::new_cell(Disjunctions::new()),
      wmc_with_disjunctions: false,
    }
  }
}

impl<P: PointerFamily> Clone for TopKProofsProvenance<P> {
  fn clone(&self) -> Self {
    Self {
      k: self.k,
      probs: P::clone_cell(&self.probs),
      disjunctions: P::clone_cell(&self.disjunctions),
      wmc_with_disjunctions: self.wmc_with_disjunctions,
    }
  }
}

impl<P: PointerFamily> TopKProofsProvenance<P> {
  pub fn new(k: usize, wmc_with_disjunctions: bool) -> Self {
    Self {
      k,
      probs: P::new_cell(Vec::new()),
      disjunctions: P::new_cell(Disjunctions::new()),
      wmc_with_disjunctions,
    }
  }

  pub fn num_facts(&self) -> usize {
    P::get_cell(&self.probs, |p| p.len())
  }

  pub fn set_k(&mut self, k: usize) {
    self.k = k;
  }

  pub fn cmp(x: &f64, y: &f64) -> bool {
    (x - y).abs() < 0.001
  }
}

impl<P: PointerFamily> DNFContextTrait for TopKProofsProvenance<P> {
  fn fact_probability(&self, id: &usize) -> f64 {
    P::get_cell(&self.probs, |p| p[*id])
  }

  fn has_disjunction_conflict(&self, pos_facts: &std::collections::BTreeSet<usize>) -> bool {
    P::get_cell(&self.disjunctions, |d| d.has_conflict(pos_facts))
  }
}

impl<P: PointerFamily> Provenance for TopKProofsProvenance<P> {
  type Tag = DNFFormula;

  type InputTag = InputExclusiveProb;

  type OutputTag = f64;

  fn name(&self) -> String {
    format!("top-k-proofs")
  }

  fn tagging_fn(&self, input_tag: Self::InputTag) -> Self::Tag {
    // First generate id and push the probability into the list
    let fact_id = self.num_facts();
    P::get_cell_mut(&self.probs, |p| p.push(input_tag.prob));

    // Add exlusion if needed
    if let Some(disj_id) = input_tag.exclusion {
      P::get_cell_mut(&self.disjunctions, |d| d.add_disjunction(disj_id, fact_id));
    }

    // Lastly return a tag
    Self::Tag::singleton(fact_id)
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    let s = RealSemiring;
    let v = |i: &usize| -> f64 { self.fact_probability(i) };
    if self.wmc_with_disjunctions {
      P::get_cell(&self.disjunctions, |disj| t.wmc_with_disjunctions(&s, &v, disj))
    } else {
      t.wmc(&s, &v)
    }
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
    self.top_k_add(t1, t2, self.k)
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    t_old == t_new
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    self.top_k_mult(t1, t2, self.k)
  }

  fn negate(&self, t: &Self::Tag) -> Option<Self::Tag> {
    Some(self.top_k_negate(t, self.k))
  }

  fn weight(&self, t: &Self::Tag) -> f64 {
    let s = RealSemiring;
    let v = |i: &usize| -> f64 { self.fact_probability(i) };
    t.wmc(&s, &v)
  }
}
