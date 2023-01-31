use itertools::Itertools;

use super::*;
use crate::common::element::*;
use crate::runtime::dynamic::*;
use crate::runtime::statics::*;
use crate::utils::PointerFamily;

pub struct DiffNandMultProbProvenance<T: Clone, P: PointerFamily> {
  pub warned_disjunction: bool,
  pub valid_threshold: f64,
  pub storage: P::Pointer<Vec<T>>,
}

impl<T: Clone, P: PointerFamily> Clone for DiffNandMultProbProvenance<T, P> {
  fn clone(&self) -> Self {
    Self {
      warned_disjunction: self.warned_disjunction,
      valid_threshold: self.valid_threshold,
      storage: P::new((&*self.storage).clone()),
    }
  }
}

impl<T: Clone + 'static, P: PointerFamily> DiffNandMultProbProvenance<T, P> {
  pub fn input_tags(&self) -> Vec<T> {
    self.storage.iter().cloned().collect()
  }

  pub fn tag_of_chosen_set<E>(&self, all: &Vec<E>, chosen_ids: &Vec<usize>) -> DualNumber2
  where
    E: Element<Self>,
  {
    all
      .iter()
      .enumerate()
      .map(|(id, elem)| {
        if chosen_ids.contains(&id) {
          elem.tag().clone()
        } else {
          self.negate(&elem.tag()).unwrap()
        }
      })
      .fold(self.one(), |a, b| self.mult(&a, &b))
  }
}

impl<T: Clone, P: PointerFamily> Default for DiffNandMultProbProvenance<T, P> {
  fn default() -> Self {
    Self {
      warned_disjunction: false,
      valid_threshold: 0.0000,
      storage: P::new(Vec::new()),
    }
  }
}

impl<T: Clone + 'static, P: PointerFamily> Provenance for DiffNandMultProbProvenance<T, P> {
  type Tag = DualNumber2;

  type InputTag = InputDiffProb<T>;

  type OutputTag = OutputDiffProb<T>;

  fn name() -> &'static str {
    "diffnandmultprob"
  }

  fn tagging_fn(&mut self, input_tag: Self::InputTag) -> Self::Tag {
    let InputDiffProb(p, t) = input_tag;
    let pos_id = self.storage.len();
    P::get_mut(&mut self.storage).push(t);
    DualNumber2::new(pos_id, p)
  }

  fn recover_fn(&self, p: &Self::Tag) -> Self::OutputTag {
    let prob = p.real;
    let deriv = p
      .gradient
      .indices
      .iter()
      .zip(p.gradient.values.iter())
      .map(|(i, v)| (*i, *v, self.storage[*i].clone()))
      .collect::<Vec<_>>();
    OutputDiffProb(prob, deriv)
  }

  fn discard(&self, t: &Self::Tag) -> bool {
    t.real <= self.valid_threshold
  }

  fn zero(&self) -> Self::Tag {
    DualNumber2::zero()
  }

  fn one(&self) -> Self::Tag {
    DualNumber2::one()
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    -&(&-t1 * &-t2)
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    t1 * t2
  }

  fn negate(&self, t: &Self::Tag) -> Option<Self::Tag> {
    Some(-t)
  }

  fn saturated(&self, _: &Self::Tag, _: &Self::Tag) -> bool {
    true
  }

  fn weight(&self, t: &Self::Tag) -> f64 {
    t.real
  }

  fn dynamic_count(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let mut result = vec![];
    if batch.is_empty() {
      result.push(DynamicElement::new(0usize, self.one()));
    } else {
      for chosen_set in (0..batch.len()).powerset() {
        let count = chosen_set.len();
        let tag = self.tag_of_chosen_set(&batch, &chosen_set);
        result.push(DynamicElement::new(count, tag));
      }
    }
    result
  }

  fn dynamic_exists(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let mut max_prob = 0.0;
    let mut max_info = None;
    for elem in batch {
      let prob = elem.tag.real;
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

  fn static_count<Tup: StaticTupleTrait>(&self, batch: StaticElements<Tup, Self>) -> StaticElements<usize, Self> {
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

  fn static_exists<Tup: StaticTupleTrait>(&self, batch: StaticElements<Tup, Self>) -> StaticElements<bool, Self> {
    let mut max_prob = 0.0;
    let mut max_info = None;
    for elem in batch {
      let prob = elem.tag.real;
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
}
