use crate::common::element::*;
use crate::common::foreign_aggregate::*;
use crate::common::foreign_aggregates::*;

use crate::runtime::dynamic::*;
use crate::runtime::env::*;

use super::*;

#[derive(Clone, Debug)]
pub struct MinMaxProbProvenance {
  valid_threshold: f64,
}

impl Default for MinMaxProbProvenance {
  fn default() -> Self {
    Self {
      valid_threshold: 0.0000,
    }
  }
}

impl MinMaxProbProvenance {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn cmp(x: &f64, y: &f64) -> bool {
    (x - y).abs() < 0.001
  }
}

impl Provenance for MinMaxProbProvenance {
  type Tag = f64;

  type InputTag = f64;

  type OutputTag = f64;

  fn name(&self) -> String {
    "minmaxprob".to_string()
  }

  fn tagging_fn(&self, p: Self::InputTag) -> Self::Tag {
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
    t1.max(*t2)
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    t_old == t_new
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    t1.min(*t2)
  }

  fn negate(&self, p: &Self::Tag) -> Option<Self::Tag> {
    Some(1.0 - p)
  }

  fn weight(&self, t: &Self::Tag) -> f64 {
    *t
  }
}

impl Aggregator<MinMaxProbProvenance> for CountAggregator {
  fn aggregate(
    &self,
    p: &MinMaxProbProvenance,
    _env: &RuntimeEnvironment,
    mut batch: DynamicElements<MinMaxProbProvenance>,
  ) -> DynamicElements<MinMaxProbProvenance> {
    if self.non_multi_world {
      vec![DynamicElement::new(batch.len(), p.one())]
    } else {
      if batch.is_empty() {
        vec![DynamicElement::new(0usize, p.one())]
      } else {
        batch.sort_by(|a, b| b.tag.total_cmp(&a.tag));
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

fn max_min_prob_of_k_count<E>(sorted_set: &Vec<E>, k: usize) -> f64
where
  E: Element<MinMaxProbProvenance>,
{
  let prob = sorted_set
    .iter()
    .enumerate()
    .map(|(id, elem)| if id < k { *elem.tag() } else { 1.0 - *elem.tag() })
    .fold(f64::INFINITY, |a, b| a.min(b));
  prob.into()
}
