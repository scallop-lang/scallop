use std::marker::PhantomData;

use itertools::Itertools;

use super::*;
use crate::runtime::dynamic::*;
use crate::utils::{PointerFamily, RcFamily};

#[derive(Clone, PartialEq)]
pub struct Proofs<P: PointerFamily> {
  pub formula: DNFFormula,
  pub phantom: PhantomData<P>,
}

impl<P: PointerFamily> From<DNFFormula> for Proofs<P> {
  fn from(formula: DNFFormula) -> Self {
    Self {
      formula,
      phantom: PhantomData,
    }
  }
}

impl<P: PointerFamily> std::fmt::Debug for Proofs<P> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.formula.fmt(f)
  }
}

impl<P: PointerFamily> std::fmt::Display for Proofs<P> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.formula.fmt(f)
  }
}

impl<P: PointerFamily> Tag for Proofs<P> {
  type Context = TopKProofsContext<P>;
}

pub struct TopKProofsContext<P: PointerFamily = RcFamily> {
  pub k: usize,
  pub probs: P::Pointer<Vec<f64>>,
  pub disjunctions: Disjunctions,
}

impl<P: PointerFamily> Default for TopKProofsContext<P> {
  fn default() -> Self {
    Self {
      k: 3,
      probs: P::new(Vec::new()),
      disjunctions: Disjunctions::new(),
    }
  }
}

impl<P: PointerFamily> Clone for TopKProofsContext<P> {
  fn clone(&self) -> Self {
    Self {
      k: self.k,
      probs: P::new((&*self.probs).clone()),
      disjunctions: self.disjunctions.clone(),
    }
  }
}

impl<P: PointerFamily> TopKProofsContext<P> {
  pub fn new(k: usize) -> Self {
    Self {
      k,
      probs: P::new(Vec::new()),
      disjunctions: Disjunctions::new(),
    }
  }

  pub fn set_k(&mut self, k: usize) {
    self.k = k;
  }
}

impl<P: PointerFamily> DNFContextTrait for TopKProofsContext<P> {
  fn fact_probability(&self, id: &usize) -> f64 {
    self.probs[*id]
  }

  fn has_disjunction_conflict(&self, pos_facts: &std::collections::BTreeSet<usize>) -> bool {
    self.disjunctions.has_conflict(pos_facts)
  }
}

impl<P: PointerFamily> ProvenanceContext for TopKProofsContext<P> {
  type Tag = Proofs<P>;

  type InputTag = f64;

  type OutputTag = f64;

  fn name() -> &'static str {
    "top-k-proofs"
  }

  fn tagging_fn(&mut self, prob: Self::InputTag) -> Self::Tag {
    let id = self.probs.len();
    P::get_mut(&mut self.probs).push(prob);
    DNFFormula::singleton(id).into()
  }

  fn tagging_disjunction_fn(&mut self, tags: Vec<Self::InputTag>) -> Vec<Self::Tag> {
    let mut ids = vec![];

    // Add base disjunctions
    let tags = tags
      .into_iter()
      .map(|tag| {
        let id = self.probs.len();
        P::get_mut(&mut self.probs).push(tag);
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
    let s = semirings::real::RealSemiring;
    let v = |i: &usize| -> f64 { self.probs[*i] };
    t.formula.wmc(&s, &v)
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
    self.top_k_add(&t1.formula, &t2.formula, self.k).into()
  }

  fn add_with_proceeding(
    &self,
    stable_tag: &Self::Tag,
    recent_tag: &Self::Tag,
  ) -> (Self::Tag, Proceeding) {
    let new_tag = self.top_k_add(&stable_tag.formula, &recent_tag.formula, self.k);
    let proceeding = if new_tag == recent_tag.formula || new_tag == stable_tag.formula {
      Proceeding::Stable
    } else {
      Proceeding::Recent
    };
    (new_tag.into(), proceeding)
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    self.top_k_mult(&t1.formula, &t2.formula, self.k).into()
  }

  fn negate(&self, t: &Self::Tag) -> Option<Self::Tag> {
    Some(self.top_k_negate(&t.formula, self.k).into())
  }

  fn dynamic_count(&self, op: &DynamicCountOp, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let vec_batch = project_batch_helper(batch, &op.key, self);
    if vec_batch.is_empty() {
      vec![DynamicElement::new(0usize.into(), self.one())]
    } else {
      let mut elems = vec![];
      for chosen_set in (0..vec_batch.len()).powerset() {
        let count = chosen_set.len();
        let tag = self.top_k_tag_of_chosen_set(
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
        let prob = self.top_k_tag_of_chosen_set(
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
        let prob = self.top_k_tag_of_chosen_set(
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
        &self.top_k_negate(&elem.tag.formula, self.k).into(),
      );
    }
    let t = DynamicElement::new(true.into(), exists_tag);
    let f = DynamicElement::new(false.into(), not_exists_tag);
    vec![t, f]
  }
}
