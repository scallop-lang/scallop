use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

use super::super::batching::*;
use super::*;

pub struct DynamicAggregationDataflow<'a, T: Tag> {
  pub source: DynamicGroups<'a, T>,
  pub aggregator: DynamicAggregateOp,
  pub ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for DynamicAggregationDataflow<'a, T> {
  fn clone(&self) -> Self {
    Self {
      source: self.source.clone(),
      aggregator: self.aggregator.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, T: Tag> DynamicAggregationDataflow<'a, T> {
  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::Empty
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    let agg_batch =
      DynamicBatch::aggregate(self.source.iter_groups(), self.aggregator.clone(), self.ctx);
    DynamicBatches::single(agg_batch)
  }
}

pub struct DynamicAggregationBatch<'a, T: Tag> {
  pub source: Box<DynamicGroupsIterator<T>>,
  pub aggregator: DynamicAggregateOp,
  pub curr_result: Option<std::vec::IntoIter<DynamicElement<T>>>,
  pub ctx: &'a T::Context,
}

impl<'a, T: Tag> DynamicAggregationBatch<'a, T> {
  pub fn new(
    mut source: DynamicGroupsIterator<T>,
    aggregator: DynamicAggregateOp,
    ctx: &'a T::Context,
  ) -> Self {
    let curr_group = source.next();
    let curr_result = curr_group.map(|g| g.process(&aggregator, ctx).into_iter());
    Self {
      source: Box::new(source),
      aggregator,
      curr_result,
      ctx,
    }
  }
}

impl<'a, T: Tag> Clone for DynamicAggregationBatch<'a, T> {
  fn clone(&self) -> Self {
    Self {
      source: self.source.clone(),
      aggregator: self.aggregator.clone(),
      curr_result: self.curr_result.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, T: Tag> Iterator for DynamicAggregationBatch<'a, T> {
  type Item = DynamicElement<T>;

  fn next(&mut self) -> Option<Self::Item> {
    while let Some(curr_result_iter) = &mut self.curr_result {
      if let Some(next_result) = curr_result_iter.next() {
        return Some(next_result);
      } else {
        self.curr_result = self
          .source
          .next()
          .map(|g| g.process(&self.aggregator, self.ctx).into_iter());
      }
    }
    None
  }
}
