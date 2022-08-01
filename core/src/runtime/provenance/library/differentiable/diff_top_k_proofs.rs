use std::marker::PhantomData;

use itertools::Itertools;

use super::*;
use crate::runtime::dynamic::*;
use crate::runtime::statics::*;
use crate::utils::PointerFamily;

#[derive(Clone)]
pub struct Proofs<T: Clone, P: PointerFamily> {
  pub formula: DNFFormula,
  pub phantom: PhantomData<(T, P)>,
}

impl<T: Clone, P: PointerFamily> From<DNFFormula> for Proofs<T, P> {
  fn from(formula: DNFFormula) -> Self {
    Self {
      formula,
      phantom: PhantomData,
    }
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

impl<T: Clone + 'static, P: PointerFamily> Tag for Proofs<T, P> {
  type Context = DiffTopKProofsContext<T, P>;
}

pub struct DiffTopKProofsContext<T: Clone, P: PointerFamily> {
  pub k: usize,
  pub diff_probs: P::Pointer<Vec<(f64, T)>>,
  pub disjunctions: Disjunctions,
}

impl<T: Clone, P: PointerFamily> Clone for DiffTopKProofsContext<T, P> {
  fn clone(&self) -> Self {
    Self {
      k: self.k,
      diff_probs: P::new((&*self.diff_probs).clone()),
      disjunctions: self.disjunctions.clone(),
    }
  }
}

impl<T: Clone, P: PointerFamily> DiffTopKProofsContext<T, P> {
  pub fn new(k: usize) -> Self {
    Self {
      k,
      diff_probs: P::new(Vec::new()),
      disjunctions: Disjunctions::new(),
    }
  }

  pub fn input_tags(&self) -> Vec<T> {
    self.diff_probs.iter().map(|(_, t)| t.clone()).collect()
  }

  pub fn set_k(&mut self, k: usize) {
    self.k = k;
  }
}

impl<T: Clone, P: PointerFamily> DNFContextTrait for DiffTopKProofsContext<T, P> {
  fn fact_probability(&self, id: &usize) -> f64 {
    self.diff_probs[*id].0
  }

  fn has_disjunction_conflict(&self, pos_facts: &std::collections::BTreeSet<usize>) -> bool {
    self.disjunctions.has_conflict(pos_facts)
  }
}

impl<T: Clone + 'static, P: PointerFamily> ProvenanceContext for DiffTopKProofsContext<T, P> {
  type Tag = Proofs<T, P>;

  type InputTag = InputDiffProb<T>;

  type OutputTag = OutputDiffProb<T>;

  fn name() -> &'static str {
    "diff-top-k-proofs"
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
    self.top_k_add(&t1.formula, &t2.formula, self.k).into()
  }

  fn add_with_proceeding(&self, stable_tag: &Self::Tag, recent_tag: &Self::Tag) -> (Self::Tag, Proceeding) {
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

  fn dynamic_count(&self, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    if batch.is_empty() {
      vec![DynamicElement::new(0usize, self.one())]
    } else {
      let mut elems = vec![];
      for chosen_set in (0..batch.len()).powerset() {
        let count = chosen_set.len();
        let tag = self.top_k_tag_of_chosen_set(batch.iter().map(|e| &e.tag.formula), &chosen_set, self.k);
        elems.push(DynamicElement::new(count, tag.into()));
      }
      elems
    }
  }

  fn dynamic_min(&self, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut elems = vec![];
    for i in 0..batch.len() {
      let min_elem = batch[i].tuple.clone();
      let mut agg_tag = self.one();
      for j in 0..i {
        agg_tag = self.mult(&agg_tag, &self.negate(&batch[j].tag).unwrap());
      }
      agg_tag = self.mult(&agg_tag, &batch[i].tag);
      elems.push(DynamicElement::new(min_elem, agg_tag));
    }
    elems
  }

  fn dynamic_max(&self, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut elems = vec![];
    for i in 0..batch.len() {
      let max_elem = batch[i].tuple.clone();
      let mut agg_tag = batch[i].tag.clone();
      for j in i + 1..batch.len() {
        agg_tag = self.mult(&agg_tag, &self.negate(&batch[j].tag).unwrap());
      }
      elems.push(DynamicElement::new(max_elem, agg_tag));
    }
    elems
  }

  fn dynamic_exists(&self, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut exists_tag = self.zero();
    let mut not_exists_tag = self.one();
    for elem in batch {
      exists_tag = self.add(&exists_tag, &elem.tag);
      not_exists_tag = self.mult(&not_exists_tag, &self.negate(&elem.tag).unwrap());
    }
    let t = DynamicElement::new(true, exists_tag);
    let f = DynamicElement::new(false, not_exists_tag);
    vec![t, f]
  }

  fn static_count<Tup: StaticTupleTrait>(
    &self,
    batch: StaticElements<Tup, Self::Tag>,
  ) -> StaticElements<usize, Self::Tag> {
    if batch.is_empty() {
      vec![StaticElement::new(0, self.one())]
    } else {
      let mut elems = vec![];
      for chosen_set in (0..batch.len()).powerset() {
        let count = chosen_set.len();
        let tag = self.top_k_tag_of_chosen_set(batch.iter().map(|e| &e.tag.formula), &chosen_set, self.k);
        elems.push(StaticElement::new(count, tag.into()));
      }
      elems
    }
  }

  fn static_min<Tup: StaticTupleTrait>(&self, batch: StaticElements<Tup, Self::Tag>) -> StaticElements<Tup, Self::Tag> {
    let mut elems = vec![];
    for i in 0..batch.len() {
      let min_elem = batch[i].tuple.get().clone();
      let mut agg_tag = self.one();
      for j in 0..i {
        agg_tag = self.mult(&agg_tag, &self.negate(&batch[j].tag).unwrap());
      }
      agg_tag = self.mult(&agg_tag, &batch[i].tag);
      elems.push(StaticElement::new(min_elem, agg_tag));
    }
    elems
  }

  fn static_max<Tup: StaticTupleTrait>(&self, batch: StaticElements<Tup, Self::Tag>) -> StaticElements<Tup, Self::Tag> {
    let mut elems = vec![];
    for i in 0..batch.len() {
      let max_elem = batch[i].tuple.get().clone();
      let mut agg_tag = batch[i].tag.clone();
      for j in i + 1..batch.len() {
        agg_tag = self.mult(&agg_tag, &self.negate(&batch[j].tag).unwrap());
      }
      elems.push(StaticElement::new(max_elem, agg_tag));
    }
    elems
  }

  fn static_exists<Tup: StaticTupleTrait>(
    &self,
    batch: StaticElements<Tup, Self::Tag>,
  ) -> StaticElements<bool, Self::Tag> {
    let mut exists_tag = self.zero();
    let mut not_exists_tag = self.one();
    for elem in batch {
      exists_tag = self.add(&exists_tag, &elem.tag);
      not_exists_tag = self.mult(&not_exists_tag, &self.negate(&elem.tag).unwrap());
    }
    let t = StaticElement::new(true, exists_tag);
    let f = StaticElement::new(false, not_exists_tag);
    vec![t, f]
  }
}
