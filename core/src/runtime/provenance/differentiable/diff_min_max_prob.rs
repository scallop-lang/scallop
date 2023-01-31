use itertools::Itertools;

use super::*;
use crate::common::element::*;
use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::statics::*;
use crate::utils::PointerFamily;

pub struct Prob(pub usize);

impl Prob {
  fn new(id: usize) -> Self {
    Self(id)
  }
}

impl Clone for Prob {
  fn clone(&self) -> Self {
    Self(self.0)
  }
}

impl std::fmt::Display for Prob {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{}", self.0))
  }
}

impl std::fmt::Debug for Prob {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{:?}", self.0))
  }
}

impl Tag for Prob {}

pub struct DiffMinMaxProbProvenance<T: Clone, P: PointerFamily> {
  pub warned_disjunction: bool,
  pub valid_threshold: f64,
  pub zero_index: usize,
  pub one_index: usize,
  pub diff_probs: P::Pointer<Vec<(f64, Derivative<T>)>>,
  pub negates: Vec<usize>,
}

impl<T: Clone, P: PointerFamily> Clone for DiffMinMaxProbProvenance<T, P> {
  fn clone(&self) -> Self {
    Self {
      warned_disjunction: self.warned_disjunction,
      valid_threshold: self.valid_threshold,
      zero_index: self.zero_index,
      one_index: self.one_index,
      diff_probs: P::new((&*self.diff_probs).clone()),
      negates: self.negates.clone(),
    }
  }
}

impl<T: Clone + 'static, P: PointerFamily> DiffMinMaxProbProvenance<T, P> {
  pub fn probability(&self, id: usize) -> f64 {
    self.diff_probs[id].0
  }

  pub fn collect_chosen_elements<'a, E>(&self, all: &'a Vec<E>, chosen_ids: &Vec<usize>) -> Vec<&'a E>
  where
    E: Element<Self>,
  {
    all
      .iter()
      .enumerate()
      .filter(|(i, _)| chosen_ids.contains(i))
      .map(|(_, e)| e.clone())
      .collect::<Vec<_>>()
  }

  pub fn min_tag_of_chosen_set<E: Element<Self>>(&self, all: &Vec<E>, chosen_ids: &Vec<usize>) -> Prob {
    all
      .iter()
      .enumerate()
      .map(|(id, elem)| {
        if chosen_ids.contains(&id) {
          elem.tag().clone()
        } else {
          self.negate(elem.tag()).unwrap()
        }
      })
      .fold(self.one(), |a, b| self.mult(&a, &b))
  }

  fn max_min_prob_of_k_count<E: Element<Self>>(&self, sorted_set: &Vec<E>, k: usize) -> Prob {
    sorted_set
      .iter()
      .enumerate()
      .map(|(id, elem)| {
        if id < k {
          elem.tag().clone()
        } else {
          self.negate(elem.tag()).unwrap()
        }
      })
      .fold(self.one(), |a, b| self.mult(&a, &b))
  }
}

impl<T: Clone, P: PointerFamily> Default for DiffMinMaxProbProvenance<T, P> {
  fn default() -> Self {
    let mut diff_probs = Vec::new();
    diff_probs.push((0.0, Derivative::Zero));
    diff_probs.push((1.0, Derivative::Zero));
    let mut negates = Vec::new();
    negates.push(1);
    negates.push(0);
    Self {
      warned_disjunction: false,
      valid_threshold: -0.0001,
      zero_index: 0,
      one_index: 1,
      diff_probs: P::new(diff_probs),
      negates,
    }
  }
}

#[derive(Clone)]
pub enum Derivative<T: Clone> {
  Pos(T),
  Zero,
  Neg(T),
}

#[derive(Clone)]
pub struct OutputDiffProb<T: Clone + 'static>(pub f64, pub usize, pub i32, pub Option<T>);

impl<T: Clone + 'static> std::fmt::Debug for OutputDiffProb<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_tuple("").field(&self.0).field(&self.1).field(&self.2).finish()
  }
}

impl<T: Clone + 'static> std::fmt::Display for OutputDiffProb<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_tuple("").field(&self.0).field(&self.1).field(&self.2).finish()
  }
}

impl<T: Clone + 'static, P: PointerFamily> Provenance for DiffMinMaxProbProvenance<T, P> {
  type Tag = Prob;

  type InputTag = InputDiffProb<T>;

  type OutputTag = OutputDiffProb<T>;

  fn name() -> &'static str {
    "diffminmaxprob"
  }

  fn tagging_fn(&mut self, input_tag: Self::InputTag) -> Self::Tag {
    let InputDiffProb(p, t) = input_tag;
    let pos_id = self.diff_probs.len();
    let neg_id = pos_id + 1;
    P::get_mut(&mut self.diff_probs).extend(vec![
      (p, Derivative::Pos(t.clone())),
      (1.0 - p, Derivative::Neg(t.clone())),
    ]);
    self.negates.push(neg_id);
    self.negates.push(pos_id);
    Self::Tag::new(pos_id)
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    let (p, der) = &self.diff_probs[t.0];
    match der {
      Derivative::Pos(s) => OutputDiffProb(*p, t.0, 1, Some(s.clone())),
      Derivative::Zero => OutputDiffProb(*p, 0, 0, None),
      Derivative::Neg(s) => OutputDiffProb(*p, t.0, -1, Some(s.clone())),
    }
  }

  fn discard(&self, p: &Self::Tag) -> bool {
    self.probability(p.0) <= self.valid_threshold
  }

  fn zero(&self) -> Self::Tag {
    Self::Tag::new(self.zero_index)
  }

  fn one(&self) -> Self::Tag {
    Self::Tag::new(self.one_index)
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    if self.probability(t1.0) > self.probability(t2.0) {
      t1.clone()
    } else {
      t2.clone()
    }
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    self.probability(t_old.0) == self.probability(t_new.0)
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    if self.probability(t1.0) > self.probability(t2.0) {
      t2.clone()
    } else {
      t1.clone()
    }
  }

  fn negate(&self, p: &Self::Tag) -> Option<Self::Tag> {
    Some(Self::Tag::new(self.negates[p.0]))
  }

  fn weight(&self, t: &Self::Tag) -> f64 {
    self.probability(t.0)
  }

  fn dynamic_count(&self, mut batch: DynamicElements<Self>) -> DynamicElements<Self> {
    if batch.is_empty() {
      vec![DynamicElement::new(0usize, self.one())]
    } else {
      batch.sort_by(|a, b| self.probability(b.tag.0).total_cmp(&self.probability(a.tag.0)));
      let mut elems = vec![];
      for k in 0..=batch.len() {
        let prob = self.max_min_prob_of_k_count(&batch, k);
        elems.push(DynamicElement::new(k, prob));
      }
      elems
    }
  }

  fn dynamic_sum(&self, ty: &ValueType, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let mut elems = vec![];
    for chosen_set in (0..batch.len()).powerset() {
      let chosen_elements = self.collect_chosen_elements(&batch, &chosen_set);
      let sum = ty.sum(chosen_elements.iter_tuples());
      let prob = self.min_tag_of_chosen_set(&batch, &chosen_set);
      elems.push(DynamicElement::new(sum, prob));
    }
    elems
  }

  fn dynamic_prod(&self, ty: &ValueType, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let mut elems = vec![];
    for chosen_set in (0..batch.len()).powerset() {
      let chosen_elements = self.collect_chosen_elements(&batch, &chosen_set);
      let sum = ty.prod(chosen_elements.iter_tuples());
      let prob = self.min_tag_of_chosen_set(&batch, &chosen_set);
      elems.push(DynamicElement::new(sum, prob));
    }
    elems
  }

  fn dynamic_min(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
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

  fn dynamic_max(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
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

  fn dynamic_exists(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let mut max_prob = 0.0;
    let mut max_id = None;
    for elem in batch {
      let prob = self.probability(elem.tag.0);
      if prob > max_prob {
        max_prob = prob;
        max_id = Some(elem.tag.0);
      }
    }
    if let Some(id) = max_id {
      let t = DynamicElement::new(true, Self::Tag::new(id));
      let f = DynamicElement::new(false, Self::Tag::new(self.negates[id]));
      vec![t, f]
    } else {
      vec![DynamicElement::new(false, self.one())]
    }
  }

  fn static_count<Tup: StaticTupleTrait>(&self, mut batch: StaticElements<Tup, Self>) -> StaticElements<usize, Self> {
    if batch.is_empty() {
      vec![StaticElement::new(0usize, self.one())]
    } else {
      batch.sort_by(|a, b| self.probability(b.tag.0).total_cmp(&self.probability(a.tag.0)));
      let mut elems = vec![];
      for k in 0..=batch.len() {
        let prob = self.max_min_prob_of_k_count(&batch, k);
        elems.push(StaticElement::new(k, prob));
      }
      elems
    }
  }

  fn static_sum<Tup: StaticTupleTrait + SumType>(&self, batch: StaticElements<Tup, Self>) -> StaticElements<Tup, Self> {
    let mut elems = vec![];
    for chosen_set in (0..batch.len()).powerset() {
      let chosen_elements = self.collect_chosen_elements(&batch, &chosen_set);
      let sum = Tup::sum(chosen_elements.iter_tuples().cloned());
      let prob = self.min_tag_of_chosen_set(&batch, &chosen_set);
      elems.push(StaticElement::new(sum, prob));
    }
    elems
  }

  fn static_prod<Tup: StaticTupleTrait + ProdType>(
    &self,
    batch: StaticElements<Tup, Self>,
  ) -> StaticElements<Tup, Self> {
    let mut elems = vec![];
    for chosen_set in (0..batch.len()).powerset() {
      let chosen_elements = self.collect_chosen_elements(&batch, &chosen_set);
      let prod = Tup::prod(chosen_elements.iter_tuples().cloned());
      let prob = self.min_tag_of_chosen_set(&batch, &chosen_set);
      elems.push(StaticElement::new(prod, prob));
    }
    elems
  }

  fn static_min<Tup: StaticTupleTrait>(&self, batch: StaticElements<Tup, Self>) -> StaticElements<Tup, Self> {
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

  fn static_max<Tup: StaticTupleTrait>(&self, batch: StaticElements<Tup, Self>) -> StaticElements<Tup, Self> {
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

  fn static_exists<Tup: StaticTupleTrait>(&self, batch: StaticElements<Tup, Self>) -> StaticElements<bool, Self> {
    let mut max_prob = 0.0;
    let mut max_id = None;
    for elem in batch {
      let prob = self.probability(elem.tag.0);
      if prob > max_prob {
        max_prob = prob;
        max_id = Some(elem.tag.0);
      }
    }
    if let Some(id) = max_id {
      let t = StaticElement::new(true, Self::Tag::new(id));
      let f = StaticElement::new(false, Self::Tag::new(self.negates[id]));
      vec![t, f]
    } else {
      vec![StaticElement::new(false, self.one())]
    }
  }
}
