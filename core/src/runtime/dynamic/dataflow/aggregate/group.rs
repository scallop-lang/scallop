use itertools::iproduct;

use crate::common::tuple::Tuple;
use crate::runtime::provenance::*;

use super::super::*;

#[derive(Clone)]
pub enum DynamicGroup<T: Tag> {
  Batch {
    batch: Vec<DynamicElement<T>>,
  },
  KeyedBatch {
    key: Tuple,
    batch: Vec<DynamicElement<T>>,
  },
  JoinKeyBatch {
    key: Tuple,
    to_join: Vec<(T, Tuple)>,
    batch: Vec<DynamicElement<T>>,
  },
}

impl<T: Tag> DynamicGroup<T> {
  pub fn process(
    self,
    aggregator: &DynamicAggregateOp,
    ctx: &T::Context,
  ) -> Vec<DynamicElement<T>> {
    match self {
      Self::Batch { batch } => ctx.dynamic_aggregate(aggregator, batch),
      Self::KeyedBatch { key, batch } => ctx
        .dynamic_aggregate(aggregator, batch)
        .into_iter()
        .map(|elem| {
          let tuple = Tuple::from((key.clone(), elem.tuple));
          DynamicElement::new(tuple, elem.tag)
        })
        .collect(),
      Self::JoinKeyBatch {
        key,
        to_join,
        batch,
      } => {
        let agg_elements = ctx.dynamic_aggregate(aggregator, batch);
        iproduct!(to_join, agg_elements)
          .into_iter()
          .map(|((to_join_tag, to_join_tup), agg_elem)| {
            let tuple = Tuple::from((key.clone(), to_join_tup, agg_elem.tuple));
            let tag = ctx.mult(&to_join_tag, &agg_elem.tag);
            DynamicElement::new(tuple, tag)
          })
          .collect()
      }
    }
  }
}
