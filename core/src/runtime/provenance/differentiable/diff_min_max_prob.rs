use itertools::Itertools;

use super::*;
use crate::common::element::*;
use crate::common::tensors::*;
use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::statics::*;
use crate::utils::*;

#[derive(Clone)]
pub enum Derivative {
  Pos(usize),
  Zero,
  Neg(usize),
}

impl Derivative {
  pub fn negate(&self) -> Self {
    match self {
      Self::Pos(i) => Self::Neg(i.clone()),
      Self::Zero => Self::Zero,
      Self::Neg(i) => Self::Pos(i.clone()),
    }
  }
}

#[derive(Clone)]
pub struct Prob(pub f64, pub Derivative);

impl Prob {
  pub fn new(p: f64, d: Derivative) -> Self {
    Self(p, d)
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

pub struct DiffMinMaxProbProvenance<T: FromTensor, P: PointerFamily> {
  pub storage: P::RcCell<Vec<T>>,
  pub valid_threshold: f64,
}

impl<T: FromTensor, P: PointerFamily> Clone for DiffMinMaxProbProvenance<T, P> {
  fn clone(&self) -> Self {
    Self {
      valid_threshold: self.valid_threshold,
      storage: P::new_rc_cell(P::get_rc_cell(&self.storage, |s| s.clone())),
    }
  }
}

impl<T: FromTensor, P: PointerFamily> DiffMinMaxProbProvenance<T, P> {
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

impl<T: FromTensor, P: PointerFamily> Default for DiffMinMaxProbProvenance<T, P> {
  fn default() -> Self {
    Self {
      valid_threshold: -0.0001,
      storage: P::new_rc_cell(Vec::new()),
    }
  }
}

#[derive(Clone)]
pub struct OutputDiffProb<T: FromTensor>(pub f64, pub usize, pub i32, pub Option<T>);

impl<T: FromTensor> std::fmt::Debug for OutputDiffProb<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_tuple("").field(&self.0).field(&self.1).field(&self.2).finish()
  }
}

impl<T: FromTensor> std::fmt::Display for OutputDiffProb<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_tuple("").field(&self.0).field(&self.1).field(&self.2).finish()
  }
}

impl<T: FromTensor, P: PointerFamily> Provenance for DiffMinMaxProbProvenance<T, P> {
  type Tag = Prob;

  type InputTag = InputDiffProb<T>;

  type OutputTag = OutputDiffProb<T>;

  fn name() -> &'static str {
    "diffminmaxprob"
  }

  fn tagging_fn(&self, input_tag: Self::InputTag) -> Self::Tag {
    let InputDiffProb(p, t) = input_tag;
    if let Some(external_tag) = t {
      let fact_id = P::get_rc_cell(&self.storage, |s| s.len());
      P::get_rc_cell_mut(&self.storage, |s| s.push(external_tag));
      Self::Tag::new(p, Derivative::Pos(fact_id))
    } else {
      Self::Tag::new(p, Derivative::Zero)
    }
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    match &t.1 {
      Derivative::Pos(fact_id) => OutputDiffProb(
        t.0,
        *fact_id,
        1,
        Some(P::get_rc_cell(&self.storage, |s| s[*fact_id].clone())),
      ),
      Derivative::Zero => OutputDiffProb(t.0, 0, 0, None),
      Derivative::Neg(fact_id) => OutputDiffProb(
        t.0,
        *fact_id,
        -1,
        Some(P::get_rc_cell(&self.storage, |s| s[*fact_id].clone())),
      ),
    }
  }

  fn discard(&self, p: &Self::Tag) -> bool {
    p.0 <= self.valid_threshold
  }

  fn zero(&self) -> Self::Tag {
    Self::Tag::new(0.0, Derivative::Zero)
  }

  fn one(&self) -> Self::Tag {
    Self::Tag::new(1.0, Derivative::Zero)
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    if t1.0 > t2.0 {
      t1.clone()
    } else {
      t2.clone()
    }
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    t_old.0 == t_new.0
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    if t1.0 > t2.0 {
      t2.clone()
    } else {
      t1.clone()
    }
  }

  fn negate(&self, p: &Self::Tag) -> Option<Self::Tag> {
    Some(Self::Tag::new(1.0 - p.0, p.1.negate()))
  }

  fn weight(&self, t: &Self::Tag) -> f64 {
    t.0
  }

  fn dynamic_count(&self, mut batch: DynamicElements<Self>) -> DynamicElements<Self> {
    if batch.is_empty() {
      vec![DynamicElement::new(0usize, self.one())]
    } else {
      batch.sort_by(|a, b| b.tag.0.total_cmp(&a.tag.0));
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
    let mut max_deriv = None;
    for elem in batch {
      let prob = elem.tag.0;
      if prob > max_prob {
        max_prob = prob;
        max_deriv = Some(elem.tag.1);
      }
    }
    if let Some(deriv) = max_deriv {
      let f = DynamicElement::new(false, Self::Tag::new(1.0 - max_prob, deriv.negate()));
      let t = DynamicElement::new(true, Self::Tag::new(max_prob, deriv));
      vec![f, t]
    } else {
      vec![DynamicElement::new(false, self.one())]
    }
  }

  fn static_count<Tup: StaticTupleTrait>(&self, mut batch: StaticElements<Tup, Self>) -> StaticElements<usize, Self> {
    if batch.is_empty() {
      vec![StaticElement::new(0usize, self.one())]
    } else {
      batch.sort_by(|a, b| b.tag.0.total_cmp(&a.tag.0));
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
    let mut max_deriv = None;
    for elem in batch {
      let prob = elem.tag.0;
      if prob > max_prob {
        max_prob = prob;
        max_deriv = Some(elem.tag.1);
      }
    }
    if let Some(deriv) = max_deriv {
      let f = StaticElement::new(false, Self::Tag::new(1.0 - max_prob, deriv.negate()));
      let t = StaticElement::new(true, Self::Tag::new(max_prob, deriv));
      vec![f, t]
    } else {
      vec![StaticElement::new(false, self.one())]
    }
  }
}
