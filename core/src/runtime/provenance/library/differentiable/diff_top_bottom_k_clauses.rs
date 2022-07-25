use std::collections::*;
use std::marker::PhantomData;

use itertools::Itertools;

use super::*;
use crate::runtime::dynamic::*;
use crate::utils::{PointerFamily, RcFamily};

#[derive(Clone)]
pub struct Formula<T, P> {
  pub formula: CNFDNFFormula,
  pub phantom: PhantomData<(T, P)>,
}

impl<T, P> PartialEq for Formula<T, P> {
  fn eq(&self, other: &Self) -> bool {
    self.formula.eq(&other.formula)
  }
}

impl<T, P> From<CNFDNFFormula> for Formula<T, P> {
  fn from(f: CNFDNFFormula) -> Self {
    Self {
      formula: f,
      phantom: PhantomData,
    }
  }
}

impl<T, P> std::fmt::Debug for Formula<T, P> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.formula.fmt(f)
  }
}

impl<T, P> std::fmt::Display for Formula<T, P> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.formula.fmt(f)
  }
}

impl<T: Clone + 'static, P: PointerFamily> Tag for Formula<T, P> {
  type Context = DiffTopBottomKClausesContext<T, P>;
}

#[derive(Debug)]
pub struct DiffTopBottomKClausesContext<T: Clone + 'static, P: PointerFamily = RcFamily> {
  pub k: usize,
  pub diff_probs: P::Pointer<Vec<(f64, T)>>,
  pub disjunctions: Disjunctions,
}

impl<T: Clone + 'static, P: PointerFamily> Clone for DiffTopBottomKClausesContext<T, P> {
  fn clone(&self) -> Self {
    Self {
      k: self.k,
      diff_probs: P::new((&*self.diff_probs).clone()),
      disjunctions: self.disjunctions.clone(),
    }
  }
}

impl<T: Clone + 'static, P: PointerFamily> DiffTopBottomKClausesContext<T, P> {
  pub fn new(k: usize) -> Self {
    Self {
      k,
      diff_probs: P::new(Vec::new()),
      disjunctions: Disjunctions::new(),
    }
  }

  pub fn set_k(&mut self, k: usize) {
    self.k = k;
  }

  pub fn input_tags(&self) -> Vec<T> {
    self.diff_probs.iter().map(|(_, t)| t.clone()).collect()
  }
}

impl<T: Clone + 'static, P: PointerFamily> CNFDNFContextTrait
  for DiffTopBottomKClausesContext<T, P>
{
  fn fact_probability(&self, id: &usize) -> f64 {
    self.diff_probs[*id].0
  }

  fn has_disjunction_conflict(&self, pos_facts: &BTreeSet<usize>) -> bool {
    self.disjunctions.has_conflict(pos_facts)
  }
}

impl<T: Clone + 'static, P: PointerFamily> ProvenanceContext
  for DiffTopBottomKClausesContext<T, P>
{
  type Tag = Formula<T, P>;

  type InputTag = InputDiffProb<T>;

  type OutputTag = OutputDiffProb<T>;

  fn name() -> &'static str {
    "diff-top-bottom-k-clauses"
  }

  fn tagging_fn(&mut self, input_tag: Self::InputTag) -> Self::Tag {
    let InputDiffProb(p, t) = input_tag;
    let id = self.diff_probs.len();
    P::get_mut(&mut self.diff_probs).push((p, t));
    CNFDNFFormula::dnf_singleton(id).into()
  }

  fn tagging_disjunction_fn(&mut self, input_tags: Vec<Self::InputTag>) -> Vec<Self::Tag> {
    let mut ids = vec![];

    // Add base disjunctions
    let tags = input_tags
      .into_iter()
      .map(|tag| {
        let InputDiffProb(p, t) = tag;
        let id = self.diff_probs.len();
        P::get_mut(&mut self.diff_probs).push((p, t));
        ids.push(id);
        CNFDNFFormula::dnf_singleton(id).into()
      })
      .collect::<Vec<_>>();

    // Add disjunction
    self.disjunctions.add_disjunction(ids.clone().into_iter());

    // Return tags
    tags
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    let s = semirings::dual_number::DualNumberSemiring::new(self.diff_probs.len());
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
    t.formula.is_zero()
  }

  fn zero(&self) -> Self::Tag {
    CNFDNFFormula::dnf_zero().into()
  }

  fn one(&self) -> Self::Tag {
    CNFDNFFormula::dnf_one().into()
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    self
      .top_bottom_k_add(&t1.formula, &t2.formula, self.k)
      .into()
  }

  fn add_with_proceeding(
    &self,
    stable_tag: &Self::Tag,
    recent_tag: &Self::Tag,
  ) -> (Self::Tag, Proceeding) {
    let new_tag = self.add(stable_tag, recent_tag);
    let proceeding = if &new_tag == recent_tag || &new_tag == stable_tag {
      Proceeding::Stable
    } else {
      Proceeding::Recent
    };
    (new_tag.into(), proceeding)
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    self
      .top_bottom_k_mult(&t1.formula, &t2.formula, self.k)
      .into()
  }

  fn negate(&self, t: &Self::Tag) -> Option<Self::Tag> {
    Some(self.base_negate(&t.formula).into())
  }

  fn dynamic_count(&self, op: &DynamicCountOp, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let vec_batch = project_batch_helper(batch, &op.key, self);
    if vec_batch.is_empty() {
      vec![DynamicElement::new(0usize.into(), self.one())]
    } else {
      let mut elems = vec![];
      for chosen_set in (0..vec_batch.len()).powerset() {
        let count = chosen_set.len();
        let tag = self.top_bottom_k_tag_of_chosen_set(
          vec_batch.iter().map(|e| &e.tag.formula),
          &chosen_set,
          self.k,
        );
        elems.push(DynamicElement::new(count.into(), tag.into()));
      }
      elems
    }
  }

  fn dynamic_min(&self, op: &DynamicMinOp, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut elems = vec![];
    for chosen_set in (0..batch.len()).powerset() {
      let chosen_elements = collect_chosen_elements(&batch, &chosen_set);
      let min_values = op.min(chosen_elements);
      for v in min_values {
        let prob = self.top_bottom_k_tag_of_chosen_set(
          batch.iter().map(|e| &e.tag.formula),
          &chosen_set,
          self.k,
        );
        elems.push(DynamicElement::new(v, prob.into()));
      }
    }
    elems
  }

  fn dynamic_max(&self, op: &DynamicMaxOp, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut elems = vec![];
    for chosen_set in (0..batch.len()).powerset() {
      let chosen_elements = collect_chosen_elements(&batch, &chosen_set);
      let max_values = op.max(chosen_elements);
      for v in max_values {
        let prob = self.top_bottom_k_tag_of_chosen_set(
          batch.iter().map(|e| &e.tag.formula),
          &chosen_set,
          self.k,
        );
        elems.push(DynamicElement::new(v, prob.into()));
      }
    }
    elems
  }

  fn dynamic_exists(&self, _: &DynamicExistsOp, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut exists_tag = self.zero();
    let mut not_exists_tag = self.one();
    for elem in batch {
      exists_tag = ProvenanceContext::add(self, &exists_tag, &elem.tag);
      not_exists_tag = ProvenanceContext::mult(
        self,
        &not_exists_tag,
        &self.base_negate(&elem.tag.formula).into(),
      );
    }
    let t = DynamicElement::new(true.into(), exists_tag);
    let f = DynamicElement::new(false.into(), not_exists_tag);
    vec![t, f]
  }
}
