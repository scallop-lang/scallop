use crate::common::element::*;
use crate::common::foreign_aggregate::*;
use crate::common::foreign_aggregates::*;
use crate::common::foreign_tensor::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::utils::*;

use super::*;

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

impl std::cmp::PartialEq for Prob {
  fn eq(&self, other: &Self) -> bool {
    self.0 == other.0
  }
}

impl std::cmp::Eq for Prob {}

impl std::cmp::PartialOrd for Prob {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    self.0.partial_cmp(&other.0)
  }
}

impl std::cmp::Ord for Prob {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    self.0.total_cmp(&other.0)
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

  fn name(&self) -> String {
    format!("diffminmaxprob")
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
}

impl<T: FromTensor, P: PointerFamily> Aggregator<DiffMinMaxProbProvenance<T, P>> for CountAggregator {
  fn aggregate(
    &self,
    p: &DiffMinMaxProbProvenance<T, P>,
    _env: &RuntimeEnvironment,
    mut batch: DynamicElements<DiffMinMaxProbProvenance<T, P>>,
  ) -> DynamicElements<DiffMinMaxProbProvenance<T, P>> {
    if self.non_multi_world {
      vec![DynamicElement::new(batch.len(), p.one())]
    } else {
      if batch.is_empty() {
        vec![DynamicElement::new(0usize, p.one())]
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
  }
}

fn max_min_prob_of_k_count<T, P, E>(sorted_set: &Vec<E>, k: usize) -> Prob
where
  T: FromTensor,
  P: PointerFamily,
  E: Element<DiffMinMaxProbProvenance<T, P>>,
{
  let prob = sorted_set
    .iter()
    .enumerate()
    .map(|(id, elem)| {
      if id < k {
        elem.tag().clone()
      } else {
        Prob(1.0 - elem.tag().0, elem.tag().1.negate())
      }
    })
    .fold(Prob(f64::INFINITY, Derivative::Zero), |a, b| a.min(b));
  prob.into()
}
