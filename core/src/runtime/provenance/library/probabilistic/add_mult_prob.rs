use itertools::Itertools;

use super::*;
use crate::runtime::dynamic::*;

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
  fn tag_of_chosen_set(
    &self,
    all: &Vec<DynamicElement<Prob>>,
    chosen_ids: &Vec<usize>,
  ) -> Prob {
    all
      .iter()
      .enumerate()
      .map(|(id, elem)| {
        if chosen_ids.contains(&id) {
          elem.tag.clone()
        } else {
          self.negate(&elem.tag).unwrap()
        }
      })
      .fold(self.one(), |a, b| self.mult(&a, &b))
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

  fn dynamic_count<'a>(&self, op: &DynamicCountOp, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut result = vec![];
    let vec_batch = project_batch_helper(batch, &op.key, self);
    if vec_batch.is_empty() {
      result.push(DynamicElement::new(0usize.into(), self.one()));
    } else {
      for chosen_set in (0..vec_batch.len()).powerset() {
        let count = chosen_set.len();
        let tag = self.tag_of_chosen_set(&vec_batch, &chosen_set);
        result.push(DynamicElement::new(count.into(), tag));
      }
    }
    result
  }

  fn dynamic_exists<'a>(&self, _: &DynamicExistsOp, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
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
      let f = DynamicElement::new(false.into(), Prob(-&tag.0));
      let t = DynamicElement::new(true.into(), tag);
      vec![f, t]
    } else {
      let e = DynamicElement::new(false.into(), self.one());
      vec![e]
    }
  }

  fn dynamic_unique<'a>(&self, op: &DynamicUniqueOp, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut max_prob = 0.0;
    let mut max_info = None;
    for elem in batch {
      let prob = elem.tag.0;
      if prob > max_prob {
        max_prob = prob;
        max_info = Some((op.key.eval(&elem.tuple), elem.tag.clone()));
      }
    }
    if let Some((tuple, tag)) = max_info {
      vec![DynamicElement::new(tuple, tag.clone())]
    } else {
      vec![]
    }
  }
}
