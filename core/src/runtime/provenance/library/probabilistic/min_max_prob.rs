use itertools::Itertools;

use crate::runtime::dynamic::*;

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

  fn dynamic_count(&self, op: &DynamicCountOp, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let vec_batch = project_batch_helper(batch, &op.key, self);
    if vec_batch.is_empty() {
      vec![DynamicElement::new(0usize.into(), self.one())]
    } else {
      let mut elems = vec![];
      for chosen_set in (0..vec_batch.len()).powerset() {
        let count = chosen_set.len();
        let prob = min_prob_of_chosen_set(&vec_batch, &chosen_set);
        elems.push(DynamicElement::new(count.into(), prob));
      }
      elems
    }
  }

  fn dynamic_sum(&self, op: &DynamicSumOp, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut elems = vec![];
    for chosen_set in (0..batch.len()).powerset() {
      let chosen_elements = collect_chosen_elements(&batch, &chosen_set);
      let sum = op.sum(chosen_elements);
      let prob = min_prob_of_chosen_set(&batch, &chosen_set);
      elems.push(DynamicElement::new(sum, prob));
    }
    elems
  }

  fn dynamic_prod(&self, op: &DynamicProdOp, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut elems = vec![];
    for chosen_set in (0..batch.len()).powerset() {
      let chosen_elements = collect_chosen_elements(&batch, &chosen_set);
      let sum = op.prod(chosen_elements);
      let prob = min_prob_of_chosen_set(&batch, &chosen_set);
      elems.push(DynamicElement::new(sum, prob));
    }
    elems
  }

  fn dynamic_min(&self, op: &DynamicMinOp, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut elems = vec![];
    for chosen_set in (0..batch.len()).powerset() {
      let chosen_elements = collect_chosen_elements(&batch, &chosen_set);
      let min_values = op.min(chosen_elements);
      for v in min_values {
        let prob = min_prob_of_chosen_set(&batch, &chosen_set);
        elems.push(DynamicElement::new(v, prob));
      }
    }
    elems
  }

  fn dynamic_max(&self, op: &DynamicMaxOp, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut elems = vec![];
    for chosen_set in (0..batch.len()).powerset() {
      let chosen_elements = collect_chosen_elements(&batch, &chosen_set);
      let min_values = op.max(chosen_elements);
      for v in min_values {
        let prob = min_prob_of_chosen_set(&batch, &chosen_set);
        elems.push(DynamicElement::new(v, prob));
      }
    }
    elems
  }

  fn dynamic_exists(&self, _: &DynamicExistsOp, batch: DynamicElements<Self::Tag>) -> DynamicElements<Self::Tag> {
    let mut exists_tag = self.zero();
    let mut not_exists_tag = self.one();
    for elem in batch {
      exists_tag = self.add(&exists_tag, &elem.tag);
      not_exists_tag = self.mult(&not_exists_tag, &self.negate(&elem.tag).unwrap());
    }
    vec![DynamicElement::new(true.into(), exists_tag), DynamicElement::new(false.into(), not_exists_tag)]
  }
}

fn min_prob_of_chosen_set(all: &Vec<DynamicElement<Prob>>, chosen_ids: &Vec<usize>) -> Prob {
  let prob = all
    .iter()
    .enumerate()
    .map(|(id, elem)| {
      if chosen_ids.contains(&id) {
        elem.tag.0
      } else {
        1.0 - elem.tag.0
      }
    })
    .fold(f64::INFINITY, |a, b| a.min(b));
  prob.into()
}
