use itertools::Itertools;

use crate::common::element::*;
use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::statics::*;

use super::*;

#[derive(Clone)]
pub struct Prob(pub f64);

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

impl From<f64> for Prob {
  fn from(f: f64) -> Self {
    Self(f)
  }
}

impl Tag for Prob {
  type Context = MinMaxProbContext;
}

#[derive(Clone, Debug)]
pub struct MinMaxProbContext {
  // warned_disjunction: bool,
  valid_threshold: f64,
}

impl Default for MinMaxProbContext {
  fn default() -> Self {
    Self {
      // warned_disjunction: false,
      valid_threshold: 0.0000,
    }
  }
}

impl ProvenanceContext for MinMaxProbContext {
  type Tag = Prob;

  type InputTag = f64;

  type OutputTag = f64;

  fn name() -> &'static str {
    "minmaxprob"
  }

  fn tagging_fn(&mut self, p: Self::InputTag) -> Self::Tag {
    p.into()
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    t.0
  }

  fn discard(&self, p: &Self::Tag) -> bool {
    p.0 <= self.valid_threshold
  }

  fn zero(&self) -> Self::Tag {
    0.0.into()
  }

  fn one(&self) -> Self::Tag {
    1.0.into()
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    t1.0.max(t2.0).into()
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    t1.0.min(t2.0).into()
  }

  fn negate(&self, p: &Self::Tag) -> Option<Self::Tag> {
    Some((1.0 - p.0).into())
  }

  fn dynamic_count(&self, mut batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    if batch.is_empty() {
      vec![DynamicElement::new(0usize, self.one())]
    } else {
      batch.sort_by(|a, b| b.tag.0.total_cmp(&a.tag.0));
      let mut elems = vec![];
      for k in 0..=batch.len() {
        let prob = max_min_prob_of_k_count(&batch, k);
        elems.push(DynamicElement::new(k, prob));
      }
      elems
    }
  }

  fn dynamic_sum(&self, ty: &ValueType, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut elems = vec![];
    for chosen_set in (0..batch.len()).powerset() {
      let chosen_elements = collect_chosen_elements(&batch, &chosen_set);
      let sum = ty.sum(chosen_elements.iter_tuples());
      let prob = min_prob_of_chosen_set(&batch, &chosen_set);
      elems.push(DynamicElement::new(sum, prob));
    }
    elems
  }

  fn dynamic_prod(&self, ty: &ValueType, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut elems = vec![];
    for chosen_set in (0..batch.len()).powerset() {
      let chosen_elements = collect_chosen_elements(&batch, &chosen_set);
      let sum = ty.prod(chosen_elements.iter_tuples());
      let prob = min_prob_of_chosen_set(&batch, &chosen_set);
      elems.push(DynamicElement::new(sum, prob));
    }
    elems
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
    vec![
      DynamicElement::new(true, exists_tag),
      DynamicElement::new(false, not_exists_tag),
    ]
  }

  fn dynamic_unique(&self, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut max_prob = 0.0;
    let mut max_info = None;
    for elem in batch {
      let prob = elem.tag.0;
      if prob > max_prob {
        max_prob = prob;
        max_info = Some(elem.clone());
      }
    }
    max_info.into_iter().collect()
  }

  fn static_count<Tup: StaticTupleTrait>(
    &self,
    mut batch: StaticElements<Tup, Self::Tag>,
  ) -> StaticElements<usize, Self::Tag> {
    if batch.is_empty() {
      vec![StaticElement::new(0usize, self.one())]
    } else {
      batch.sort_by(|a, b| b.tag.0.total_cmp(&a.tag.0));
      let mut elems = vec![];
      for k in 0..=batch.len() {
        let prob = max_min_prob_of_k_count(&batch, k);
        elems.push(StaticElement::new(k, prob));
      }
      elems
    }
  }

  fn static_sum<Tup: StaticTupleTrait + SumType>(
    &self,
    batch: StaticElements<Tup, Self::Tag>,
  ) -> StaticElements<Tup, Self::Tag> {
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
    batch: StaticElements<Tup, Self::Tag>,
  ) -> StaticElements<Tup, Self::Tag> {
    let mut elems = vec![];
    for chosen_set in (0..batch.len()).powerset() {
      let chosen_elements = collect_chosen_elements(&batch, &chosen_set);
      let prod = Tup::prod(chosen_elements.iter_tuples().cloned());
      let prob = min_prob_of_chosen_set(&batch, &chosen_set);
      elems.push(StaticElement::new(prod, prob));
    }
    elems
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
    let mut max_prob = 0.0;
    let mut max_id = None;
    for elem in batch {
      let prob = elem.tag.0;
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

  fn static_unique<Tup: StaticTupleTrait>(
    &self,
    batch: StaticElements<Tup, Self::Tag>,
  ) -> StaticElements<Tup, Self::Tag> {
    let mut max_prob = 0.0;
    let mut max_info = None;
    for elem in batch {
      let prob = elem.tag.0;
      if prob > max_prob {
        max_prob = prob;
        max_info = Some(elem.clone());
      }
    }
    max_info.into_iter().collect()
  }
}

fn min_prob_of_chosen_set<E: Element<Prob>>(all: &Vec<E>, chosen_ids: &Vec<usize>) -> Prob {
  let prob = all
    .iter()
    .enumerate()
    .map(|(id, elem)| {
      if chosen_ids.contains(&id) {
        elem.tag().0
      } else {
        1.0 - elem.tag().0
      }
    })
    .fold(f64::INFINITY, |a, b| a.min(b));
  prob.into()
}

fn max_min_prob_of_k_count<E: Element<Prob>>(sorted_set: &Vec<E>, k: usize) -> Prob {
  let prob = sorted_set
    .iter()
    .enumerate()
    .map(|(id, elem)| if id < k { elem.tag().0 } else { 1.0 - elem.tag().0 })
    .fold(f64::INFINITY, |a, b| a.min(b));
  prob.into()
}
