use std::collections::*;
use std::marker::PhantomData;

use itertools::Itertools;

use super::*;
use crate::runtime::dynamic::*;
use crate::runtime::statics::*;
use crate::utils::{PointerFamily, RcFamily};

#[derive(Clone)]
pub struct Formula<P> {
  formula: CNFDNFFormula,
  phantom: PhantomData<P>,
}

impl<P> PartialEq for Formula<P> {
  fn eq(&self, other: &Self) -> bool {
    self.formula.eq(&other.formula)
  }
}

impl<P> From<CNFDNFFormula> for Formula<P> {
  fn from(f: CNFDNFFormula) -> Self {
    Self {
      formula: f,
      phantom: PhantomData,
    }
  }
}

impl<P> std::fmt::Debug for Formula<P> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.formula.fmt(f)
  }
}

impl<P> std::fmt::Display for Formula<P> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.formula.fmt(f)
  }
}

impl<P: PointerFamily> Tag for Formula<P> {
  type Context = TopBottomKClausesContext<P>;
}

#[derive(Debug)]
pub struct TopBottomKClausesContext<P: PointerFamily = RcFamily> {
  pub k: usize,
  pub probs: P::Pointer<Vec<f64>>,
  pub disjunctions: Disjunctions,
}

impl<P: PointerFamily> Clone for TopBottomKClausesContext<P> {
  fn clone(&self) -> Self {
    Self {
      k: self.k,
      probs: P::new((&*self.probs).clone()),
      disjunctions: self.disjunctions.clone(),
    }
  }
}

impl<P: PointerFamily> TopBottomKClausesContext<P> {
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

impl<P: PointerFamily> CNFDNFContextTrait for TopBottomKClausesContext<P> {
  fn fact_probability(&self, id: &usize) -> f64 {
    self.probs[*id]
  }

  fn has_disjunction_conflict(&self, pos_facts: &BTreeSet<usize>) -> bool {
    self.disjunctions.has_conflict(pos_facts)
  }
}

impl<P: PointerFamily> ProvenanceContext for TopBottomKClausesContext<P> {
  type Tag = Formula<P>;

  type InputTag = f64;

  type OutputTag = f64;

  fn name() -> &'static str {
    "top-bottom-k-clauses"
  }

  fn tagging_fn(&mut self, prob: Self::InputTag) -> Self::Tag {
    let id = self.probs.len();
    P::get_mut(&mut self.probs).push(prob);
    CNFDNFFormula::dnf_singleton(id).into()
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
        CNFDNFFormula::dnf_singleton(id).into()
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
    AsBooleanFormula::wmc(&t.formula, &s, &v)
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
    self.top_bottom_k_add(&t1.formula, &t2.formula, self.k).into()
  }

  fn add_with_proceeding(&self, stable_tag: &Self::Tag, recent_tag: &Self::Tag) -> (Self::Tag, Proceeding) {
    let new_tag = self.add(stable_tag, recent_tag);
    let proceeding = if &new_tag == recent_tag || &new_tag == stable_tag {
      Proceeding::Stable
    } else {
      Proceeding::Recent
    };
    (new_tag.into(), proceeding)
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    self.top_bottom_k_mult(&t1.formula, &t2.formula, self.k).into()
  }

  fn negate(&self, t: &Self::Tag) -> Option<Self::Tag> {
    Some(self.base_negate(&t.formula).into())
  }

  fn dynamic_count(&self, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    if batch.is_empty() {
      vec![DynamicElement::new(0usize, self.one())]
    } else {
      let mut elems = vec![];
      for chosen_set in (0..batch.len()).powerset() {
        let count = chosen_set.len();
        let tag = self.top_bottom_k_tag_of_chosen_set(batch.iter().map(|e| &e.tag.formula), &chosen_set, self.k);
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
        let tag = self.top_bottom_k_tag_of_chosen_set(batch.iter().map(|e| &e.tag.formula), &chosen_set, self.k);
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
