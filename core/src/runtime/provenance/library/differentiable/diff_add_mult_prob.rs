use std::marker::PhantomData;

use itertools::Itertools;

use super::*;
use crate::runtime::dynamic::*;
use crate::utils::PointerFamily;

#[derive(Clone, Debug)]
pub struct Derivative {
  pub indices: Vec<usize>,
  pub values: Vec<f64>,
}

impl Derivative {
  pub fn empty() -> Self {
    Self {
      indices: vec![],
      values: vec![],
    }
  }

  pub fn singleton(id: usize) -> Self {
    Self {
      indices: vec![id],
      values: vec![1.0],
    }
  }
}

impl<'a> std::ops::Mul<&'a f64> for &'a Derivative {
  type Output = Derivative;

  fn mul(self, rhs: &'a f64) -> Self::Output {
    Self::Output {
      indices: self.indices.clone(),
      values: self.values.iter().map(|v| v * rhs).collect(),
    }
  }
}

impl<'a> std::ops::Add<&'a Derivative> for Derivative {
  type Output = Self;

  fn add(self, rhs: &'a Derivative) -> Self::Output {
    let new_capacity = self.indices.len().max(rhs.indices.len());
    let mut new_indices = Vec::with_capacity(new_capacity);
    let mut new_values = Vec::with_capacity(new_capacity);

    // Iterate through linearly; making sure that the list is sorted
    let mut i = 0;
    let mut j = 0;
    loop {
      let i_in_range = i < self.indices.len();
      let j_in_range = j < rhs.indices.len();
      let both_in_range = i_in_range && j_in_range;
      if both_in_range {
        if self.indices[i] == rhs.indices[j] {
          new_indices.push(self.indices[i]);
          new_values.push(self.values[i] + rhs.values[j]);
          i += 1;
          j += 1;
        } else if self.indices[i] < rhs.indices[j] {
          new_indices.push(self.indices[i]);
          new_values.push(self.values[i]);
          i += 1;
        } else {
          new_indices.push(rhs.indices[j]);
          new_values.push(rhs.values[j]);
          j += 1;
        }
      } else if i_in_range {
        new_indices.push(self.indices[i]);
        new_values.push(self.values[i]);
        i += 1;
      } else if j_in_range {
        new_indices.push(rhs.indices[j]);
        new_values.push(rhs.values[j]);
        j += 1;
      } else {
        break;
      }
    }

    // Construct the output
    Self::Output {
      indices: new_indices,
      values: new_values,
    }
  }
}

impl<'a> std::ops::Neg for &'a Derivative {
  type Output = Derivative;

  fn neg(self) -> Self::Output {
    Self::Output {
      indices: self.indices.clone(),
      values: self.values.iter().map(|v| -v).collect(),
    }
  }
}

#[derive(Clone)]
pub struct DiffProb<T: Clone, P: PointerFamily> {
  pub prob: f64,
  pub derivative: Derivative,
  phantom: PhantomData<(T, P)>,
}

impl<T: Clone, P: PointerFamily> DiffProb<T, P> {
  pub fn new(id: usize, prob: f64) -> Self {
    Self {
      prob,
      derivative: Derivative::singleton(id),
      phantom: PhantomData,
    }
  }

  pub fn one() -> Self {
    Self {
      prob: 1.0,
      derivative: Derivative::empty(),
      phantom: PhantomData,
    }
  }

  pub fn zero() -> Self {
    Self {
      prob: 0.0,
      derivative: Derivative::empty(),
      phantom: PhantomData,
    }
  }
}

impl<'a, T: Clone, P: PointerFamily> std::ops::Neg for &'a DiffProb<T, P> {
  type Output = DiffProb<T, P>;

  fn neg(self) -> Self::Output {
    Self::Output {
      prob: 1.0 - self.prob,
      derivative: -&self.derivative,
      phantom: PhantomData,
    }
  }
}

impl<T: Clone, P: PointerFamily> std::fmt::Debug for DiffProb<T, P> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("DiffProb")
      .field("prob", &self.prob)
      .field("derivative", &self.derivative)
      .finish()
  }
}

impl<T: Clone, P: PointerFamily> std::fmt::Display for DiffProb<T, P> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{}", self.prob))
  }
}

impl<'a, T: Clone, P: PointerFamily> std::ops::Mul<&'a DiffProb<T, P>> for &'a DiffProb<T, P> {
  type Output = DiffProb<T, P>;

  fn mul(self, rhs: &'a DiffProb<T, P>) -> Self::Output {
    Self::Output {
      prob: self.prob * rhs.prob,
      derivative: &self.derivative * &rhs.prob + &(&rhs.derivative * &self.prob),
      phantom: PhantomData,
    }
  }
}

impl<'a, T: Clone, P: PointerFamily> std::ops::Add<&'a DiffProb<T, P>> for &'a DiffProb<T, P> {
  type Output = DiffProb<T, P>;

  fn add(self, rhs: &'a DiffProb<T, P>) -> Self::Output {
    let sum = self.prob + rhs.prob;
    Self::Output {
      prob: sum.min(1.0),
      derivative: self.derivative.clone() + &rhs.derivative,
      phantom: PhantomData,
    }
  }
}

impl<T: Clone + 'static, P: PointerFamily> Tag for DiffProb<T, P> {
  type Context = DiffAddMultProbContext<T, P>;
}

pub struct DiffAddMultProbContext<T: Clone, P: PointerFamily> {
  pub warned_disjunction: bool,
  pub valid_threshold: f64,
  pub storage: P::Pointer<Vec<T>>,
}

impl<T: Clone, P: PointerFamily> Clone for DiffAddMultProbContext<T, P> {
  fn clone(&self) -> Self {
    Self {
      warned_disjunction: self.warned_disjunction,
      valid_threshold: self.valid_threshold,
      storage: P::new((&*self.storage).clone()),
    }
  }
}

impl<T: Clone + 'static, P: PointerFamily> DiffAddMultProbContext<T, P> {
  pub fn input_tags(&self) -> Vec<T> {
    self.storage.iter().cloned().collect()
  }

  pub fn tag_of_chosen_set(
    &self,
    all: &Vec<DynamicElement<DiffProb<T, P>>>,
    chosen_ids: &Vec<usize>,
  ) -> DiffProb<T, P> {
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

impl<T: Clone, P: PointerFamily> Default for DiffAddMultProbContext<T, P> {
  fn default() -> Self {
    Self {
      warned_disjunction: false,
      valid_threshold: 0.0000,
      storage: P::new(Vec::new()),
    }
  }
}

impl<T: Clone + 'static, P: PointerFamily> ProvenanceContext for DiffAddMultProbContext<T, P> {
  type Tag = DiffProb<T, P>;

  type InputTag = InputDiffProb<T>;

  type OutputTag = OutputDiffProb<T>;

  fn name() -> &'static str {
    "diffaddmultprob"
  }

  fn tagging_fn(&mut self, input_tag: Self::InputTag) -> Self::Tag {
    let InputDiffProb(p, t) = input_tag;
    let pos_id = self.storage.len();
    P::get_mut(&mut self.storage).push(t);
    Self::Tag::new(pos_id, p)
  }

  fn recover_fn(&self, p: &Self::Tag) -> Self::OutputTag {
    let prob = p.prob;
    let deriv = p
      .derivative
      .indices
      .iter()
      .zip(p.derivative.values.iter())
      .map(|(i, v)| (*i, *v, self.storage[*i].clone()))
      .collect::<Vec<_>>();
    OutputDiffProb(prob, deriv)
  }

  fn discard(&self, t: &Self::Tag) -> bool {
    t.prob <= self.valid_threshold
  }

  fn zero(&self) -> Self::Tag {
    Self::Tag::zero()
  }

  fn one(&self) -> Self::Tag {
    Self::Tag::one()
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    t1 + t2
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    t1 * t2
  }

  fn negate(&self, t: &Self::Tag) -> Option<Self::Tag> {
    Some(-t)
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
      let prob = elem.tag.prob;
      if prob > max_prob {
        max_prob = prob;
        max_info = Some(elem.tag.clone());
      }
    }
    if let Some(tag) = max_info {
      let f = DynamicElement::new(false.into(), -&tag);
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
      let prob = elem.tag.prob;
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
