use super::*;
use crate::common::element::*;
use crate::common::foreign_tensor::*;
use crate::utils::PointerFamily;

pub struct DiffAddMultProbProvenance<T: FromTensor, P: PointerFamily> {
  pub valid_threshold: f64,
  pub storage: P::RcCell<Vec<T>>,
}

impl<T: FromTensor, P: PointerFamily> Clone for DiffAddMultProbProvenance<T, P> {
  fn clone(&self) -> Self {
    Self {
      valid_threshold: self.valid_threshold,
      storage: P::new_rc_cell(P::get_rc_cell(&self.storage, |s| s.clone())),
    }
  }
}

impl<T: FromTensor, P: PointerFamily> DiffAddMultProbProvenance<T, P> {
  pub fn input_tags(&self) -> Vec<T> {
    P::get_rc_cell(&self.storage, |s| s.clone())
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

impl<T: FromTensor, P: PointerFamily> Default for DiffAddMultProbProvenance<T, P> {
  fn default() -> Self {
    Self {
      valid_threshold: 0.0000,
      storage: P::new_rc_cell(Vec::new()),
    }
  }
}

impl<T: FromTensor, P: PointerFamily> Provenance for DiffAddMultProbProvenance<T, P> {
  type Tag = DualNumber2;

  type InputTag = InputDiffProb<T>;

  type OutputTag = OutputDiffProb;

  fn name() -> &'static str {
    "diffaddmultprob"
  }

  fn tagging_fn(&self, input_tag: Self::InputTag) -> Self::Tag {
    let InputDiffProb(p, t) = input_tag;
    if let Some(external_input_tag) = t {
      let pos_id = P::get_rc_cell(&self.storage, |s| s.len());
      P::get_rc_cell_mut(&self.storage, |s| s.push(external_input_tag));
      DualNumber2::new(pos_id, p)
    } else {
      DualNumber2::constant(p)
    }
  }

  fn recover_fn(&self, p: &Self::Tag) -> Self::OutputTag {
    let prob = p.real;
    let deriv = p
      .gradient
      .indices
      .iter()
      .zip(p.gradient.values.iter())
      .map(|(i, v)| (*i, *v))
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
    let mut dn = t1 + t2;
    dn.clamp_real();
    dn
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
}
