use itertools::Itertools;

use super::*;
use crate::common::element::*;
use crate::runtime::dynamic::*;
use crate::runtime::statics::*;

#[derive(Clone, Debug)]
pub struct AddMultProbContext {
  valid_threshold: f64,
}

impl AddMultProbContext {
  fn tag_of_chosen_set<E: Element<Self>>(&self, all: &Vec<E>, chosen_ids: &Vec<usize>) -> f64 {
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

  /// The soft comparison between two probabilities
  ///
  /// This function is commonly used for testing purpose
  pub fn soft_cmp(fst: &f64, snd: &f64) -> bool {
    (fst - snd).abs() < 0.001
  }
}

impl Default for AddMultProbContext {
  fn default() -> Self {
    Self {
      valid_threshold: 0.0000,
    }
  }
}

impl Provenance for AddMultProbContext {
  type Tag = f64;

  type InputTag = f64;

  type OutputTag = f64;

  fn name() -> &'static str {
    "addmultprob"
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
    (t1 + t2).min(1.0)
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    t1 * t2
  }

  fn negate(&self, p: &Self::Tag) -> Option<Self::Tag> {
    Some(1.0 - p)
  }

  fn saturated(&self, _: &Self::Tag, _: &Self::Tag) -> bool {
    true
  }

  fn dynamic_count(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    if batch.is_empty() {
      vec![DynamicElement::new(0usize, self.one())]
    } else {
      let mut result = vec![];
      for chosen_set in (0..batch.len()).powerset() {
        let count = chosen_set.len();
        let tag = self.tag_of_chosen_set(&batch, &chosen_set);
        result.push(DynamicElement::new(count, tag));
      }
      result
    }
  }

  fn dynamic_exists(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let mut max_prob = 0.0;
    let mut max_info = None;
    for elem in batch {
      let prob = elem.tag;
      if prob > max_prob {
        max_prob = prob;
        max_info = Some(elem.tag.clone());
      }
    }
    if let Some(tag) = max_info {
      let f = DynamicElement::new(false, self.negate(&tag).unwrap());
      let t = DynamicElement::new(true, tag);
      vec![f, t]
    } else {
      let e = DynamicElement::new(false, self.one());
      vec![e]
    }
  }

  fn dynamic_top_k(&self, k: usize, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let ids = aggregate_top_k_helper(batch.len(), k, |id| batch[id].tag);
    ids.into_iter().map(|id| batch[id].clone()).collect()
  }

  fn static_count<Tup: StaticTupleTrait>(
    &self,
    batch: StaticElements<Tup, Self>,
  ) -> StaticElements<usize, Self> {
    let mut result = vec![];
    if batch.is_empty() {
      result.push(StaticElement::new(0usize, self.one()));
    } else {
      for chosen_set in (0..batch.len()).powerset() {
        let count = chosen_set.len();
        let tag = self.tag_of_chosen_set(&batch, &chosen_set);
        result.push(StaticElement::new(count, tag));
      }
    }
    result
  }

  fn static_exists<Tup: StaticTupleTrait>(
    &self,
    batch: StaticElements<Tup, Self>,
  ) -> StaticElements<bool, Self> {
    let mut max_prob = 0.0;
    let mut max_info = None;
    for elem in batch {
      let prob = elem.tag;
      if prob > max_prob {
        max_prob = prob;
        max_info = Some(elem.tag.clone());
      }
    }
    if let Some(tag) = max_info {
      let f = StaticElement::new(false, self.negate(&tag).unwrap());
      let t = StaticElement::new(true, tag);
      vec![f, t]
    } else {
      let e = StaticElement::new(false, self.one());
      vec![e]
    }
  }

  fn static_top_k<Tup: StaticTupleTrait>(
    &self,
    k: usize,
    batch: StaticElements<Tup, Self>,
  ) -> StaticElements<Tup, Self> {
    let ids = aggregate_top_k_helper(batch.len(), k, |id| batch[id].tag);
    ids.into_iter().map(|id| batch[id].clone()).collect()
  }
}
