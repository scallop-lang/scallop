use itertools::Itertools;

use super::*;
use crate::common::element::*;
use crate::runtime::dynamic::*;
use crate::runtime::statics::*;

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
  type Context = AddMultProbContext;
}

#[derive(Clone, Debug)]
pub struct AddMultProbContext {
  valid_threshold: f64,
}

impl AddMultProbContext {
  fn tag_of_chosen_set<E: Element<Prob>>(&self, all: &Vec<E>, chosen_ids: &Vec<usize>) -> Prob {
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

impl ProvenanceContext for AddMultProbContext {
  type Tag = Prob;

  type InputTag = f64;

  type OutputTag = f64;

  fn name() -> &'static str {
    "addmultprob"
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
    (t1.0 + t2.0).min(1.0).into()
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    (t1.0 * t2.0).into()
  }

  fn negate(&self, p: &Self::Tag) -> Option<Self::Tag> {
    Some((1.0 - p.0).into())
  }

  fn dynamic_count(&self, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
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

  fn dynamic_exists(&self, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut max_prob = 0.0;
    let mut max_info = None;
    for elem in batch {
      let prob = elem.tag.0;
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
    batch: StaticElements<Tup, Self::Tag>,
  ) -> StaticElements<usize, Self::Tag> {
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
    batch: StaticElements<Tup, Self::Tag>,
  ) -> StaticElements<bool, Self::Tag> {
    let mut max_prob = 0.0;
    let mut max_info = None;
    for elem in batch {
      let prob = elem.tag.0;
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
