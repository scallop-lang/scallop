use itertools::Itertools;

use crate::common::element::*;
use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::statics::*;

use super::*;

#[derive(Clone, Debug)]
pub struct MinMaxProbProvenance {
  valid_threshold: f64,
}

impl Default for MinMaxProbProvenance {
  fn default() -> Self {
    Self {
      valid_threshold: 0.0000,
    }
  }
}

impl MinMaxProbProvenance {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn cmp(x: &f64, y: &f64) -> bool {
    x == y
  }
}

impl Provenance for MinMaxProbProvenance {
  type Tag = f64;

  type InputTag = f64;

  type OutputTag = f64;

  fn name() -> &'static str {
    "minmaxprob"
  }

  fn tagging_fn(&mut self, p: Self::InputTag) -> Self::Tag {
    p.into()
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    *t
  }

  fn discard(&self, p: &Self::Tag) -> bool {
    p <= &self.valid_threshold
  }

  fn zero(&self) -> Self::Tag {
    0.0
  }

  fn one(&self) -> Self::Tag {
    1.0
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    t1.max(*t2)
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    t_old == t_new
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    t1.min(*t2)
  }

  fn negate(&self, p: &Self::Tag) -> Option<Self::Tag> {
    Some(1.0 - p)
  }

  fn weight(&self, t: &Self::Tag) -> f64 {
    *t
  }

  fn dynamic_count(&self, mut batch: DynamicElements<Self>) -> DynamicElements<Self> {
    if batch.is_empty() {
      vec![DynamicElement::new(0usize, self.one())]
    } else {
      batch.sort_by(|a, b| b.tag.total_cmp(&a.tag));
      let mut elems = vec![];
      for k in 0..=batch.len() {
        let prob = max_min_prob_of_k_count(&batch, k);
        elems.push(DynamicElement::new(k, prob));
      }
      elems
    }
  }

  fn dynamic_sum(&self, ty: &ValueType, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let mut elems = vec![];
    for chosen_set in (0..batch.len()).powerset() {
      let chosen_elements = collect_chosen_elements(&batch, &chosen_set);
      let sum = ty.sum(chosen_elements.iter_tuples());
      let prob = min_prob_of_chosen_set(&batch, &chosen_set);
      elems.push(DynamicElement::new(sum, prob));
    }
    elems
  }

  fn dynamic_prod(&self, ty: &ValueType, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let mut elems = vec![];
    for chosen_set in (0..batch.len()).powerset() {
      let chosen_elements = collect_chosen_elements(&batch, &chosen_set);
      let sum = ty.prod(chosen_elements.iter_tuples());
      let prob = min_prob_of_chosen_set(&batch, &chosen_set);
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
    let mut exists_tag = self.zero();
    let mut not_exists_tag = self.one();
    for elem in batch {
      exists_tag = self.add(&exists_tag, &elem.tag);
      not_exists_tag = self.mult(&not_exists_tag, &self.negate(&elem.tag).unwrap());
    }
    vec![
      DynamicElement::new(true, exists_tag),
      DynamicElement::new(false, not_exists_tag),
    ]
  }

  fn static_count<Tup: StaticTupleTrait>(&self, mut batch: StaticElements<Tup, Self>) -> StaticElements<usize, Self> {
    if batch.is_empty() {
      vec![StaticElement::new(0usize, self.one())]
    } else {
      batch.sort_by(|a, b| b.tag.total_cmp(&a.tag));
      let mut elems = vec![];
      for k in 0..=batch.len() {
        let prob = max_min_prob_of_k_count(&batch, k);
        elems.push(StaticElement::new(k, prob));
      }
      elems
    }
  }

  fn static_sum<Tup: StaticTupleTrait + SumType>(&self, batch: StaticElements<Tup, Self>) -> StaticElements<Tup, Self> {
    let mut elems = vec![];
    for chosen_set in (0..batch.len()).powerset() {
      let chosen_elements = collect_chosen_elements(&batch, &chosen_set);
      let sum = Tup::sum(chosen_elements.iter_tuples().cloned());
      let prob = min_prob_of_chosen_set(&batch, &chosen_set);
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
      let chosen_elements = collect_chosen_elements(&batch, &chosen_set);
      let prod = Tup::prod(chosen_elements.iter_tuples().cloned());
      let prob = min_prob_of_chosen_set(&batch, &chosen_set);
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
      let prob = elem.tag;
      if prob > max_prob {
        max_prob = prob;
        max_id = Some(elem.tag.clone());
      }
    }
    if let Some(tag) = max_id {
      let f = StaticElement::new(false, self.negate(&tag).unwrap());
      let t = StaticElement::new(true, tag);
      vec![t, f]
    } else {
      vec![StaticElement::new(false, self.one())]
    }
  }
}

fn min_prob_of_chosen_set<E>(all: &Vec<E>, chosen_ids: &Vec<usize>) -> f64
where
  E: Element<MinMaxProbProvenance>,
{
  let prob = all
    .iter()
    .enumerate()
    .map(|(id, elem)| {
      if chosen_ids.contains(&id) {
        *elem.tag()
      } else {
        1.0 - *elem.tag()
      }
    })
    .fold(f64::INFINITY, |a, b| a.min(b));
  prob.into()
}

fn max_min_prob_of_k_count<E>(sorted_set: &Vec<E>, k: usize) -> f64
where
  E: Element<MinMaxProbProvenance>,
{
  let prob = sorted_set
    .iter()
    .enumerate()
    .map(|(id, elem)| if id < k { *elem.tag() } else { 1.0 - *elem.tag() })
    .fold(f64::INFINITY, |a, b| a.min(b));
  prob.into()
}
