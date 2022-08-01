use std::marker::PhantomData;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

use super::super::*;

pub struct AggregationImplicitGroup<'a, A, D, K, T1, T>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  D: Dataflow<(K, T1), T>,
  A: Aggregator<T1, T>,
  T: Tag,
{
  agg: A,
  d: D,
  ctx: &'a T::Context,
  phantom: PhantomData<(K, T1)>,
}

impl<'a, A, D, K, T1, T> AggregationImplicitGroup<'a, A, D, K, T1, T>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  D: Dataflow<(K, T1), T>,
  A: Aggregator<T1, T>,
  T: Tag,
{
  pub fn new(agg: A, d: D, ctx: &'a T::Context) -> Self {
    Self {
      agg,
      d,
      ctx,
      phantom: PhantomData,
    }
  }
}

impl<'a, A, D, K, T1, T> Dataflow<(K, A::Output), T> for AggregationImplicitGroup<'a, A, D, K, T1, T>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  D: Dataflow<(K, T1), T>,
  A: Aggregator<T1, T>,
  T: Tag,
{
  type Stable = EmptyBatches<std::iter::Empty<StaticElement<(K, A::Output), T>>>;

  type Recent = SingleBatch<std::vec::IntoIter<StaticElement<(K, A::Output), T>>>;

  fn iter_stable(&self) -> Self::Stable {
    Self::Stable::default()
  }

  fn iter_recent(self) -> Self::Recent {
    // Sanitize input relation
    let mut batch = if let Some(b) = self.d.iter_recent().next() {
      b
    } else {
      return Self::Recent::empty();
    };

    // Cache the context
    let agg = self.agg;
    let ctx = self.ctx;

    // Temporary function to aggregate the group and populate the result
    let consolidate_group =
      |result: &mut StaticElements<(K, A::Output), T>, agg_key: K, agg_group: StaticElements<T1, T>| {
        let agg_results = agg.aggregate(agg_group, ctx);
        let joined_results = agg_results
          .into_iter()
          .map(|agg_result| StaticElement::new((agg_key.clone(), agg_result.tuple.get().clone()), agg_result.tag));
        result.extend(joined_results);
      };

    // Get the first element from the batch; otherwise, return empty
    let first_elem = if let Some(e) = batch.next() {
      e
    } else {
      return Self::Recent::empty();
    };

    // Internal states
    let mut result = vec![];
    let mut agg_key = first_elem.tuple.0.clone();
    let mut agg_group = vec![StaticElement::new(first_elem.tuple.1.clone(), first_elem.tag.clone())];

    // Enter main loop
    while let Some(curr_elem) = batch.next() {
      let curr_key = curr_elem.tuple.0.clone();
      if curr_key == agg_key {
        // If the key is the same, add this element to the same batch
        agg_group.push(StaticElement::new(curr_elem.tuple.1.clone(), curr_elem.tag.clone()));
      } else {
        // Add the group into the results
        let mut new_agg_key = curr_elem.tuple.0.clone();
        std::mem::swap(&mut new_agg_key, &mut agg_key);
        let mut new_agg_group = vec![StaticElement::new(curr_elem.tuple.1.clone(), curr_elem.tag.clone())];
        std::mem::swap(&mut new_agg_group, &mut agg_group);
        consolidate_group(&mut result, new_agg_key, new_agg_group);
      }
    }

    // Make sure we handle the last group
    consolidate_group(&mut result, agg_key, agg_group);

    // Return the result as a single batch
    Self::Recent::singleton(result.into_iter())
  }
}

impl<'a, A, D, K, T1, T> Clone for AggregationImplicitGroup<'a, A, D, K, T1, T>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  D: Dataflow<(K, T1), T>,
  A: Aggregator<T1, T>,
  T: Tag,
{
  fn clone(&self) -> Self {
    Self {
      agg: self.agg.clone(),
      d: self.d.clone(),
      ctx: self.ctx,
      phantom: PhantomData,
    }
  }
}
