use std::marker::PhantomData;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

use super::super::*;

pub struct AggregationSingleGroup<'a, A, D, T1, T>
where
  T1: StaticTupleTrait,
  D: Dataflow<T1, T>,
  A: Aggregator<T1, T>,
  T: Tag,
{
  agg: A,
  d: D,
  ctx: &'a T::Context,
  phantom: PhantomData<T1>,
}

impl<'a, A, D, T1, T> AggregationSingleGroup<'a, A, D, T1, T>
where
  T1: StaticTupleTrait,
  D: Dataflow<T1, T>,
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

impl<'a, A, D, T1, T> Dataflow<A::Output, T> for AggregationSingleGroup<'a, A, D, T1, T>
where
  T1: StaticTupleTrait,
  D: Dataflow<T1, T>,
  A: Aggregator<T1, T>,
  T: Tag,
{
  type Stable = EmptyBatches<std::iter::Empty<StaticElement<A::Output, T>>>;

  type Recent = SingleBatch<std::vec::IntoIter<StaticElement<A::Output, T>>>;

  fn iter_stable(&self) -> Self::Stable {
    Self::Stable::default()
  }

  fn iter_recent(self) -> Self::Recent {
    // Sanitize input relation
    let batch = if let Some(b) = self.d.iter_recent().next() {
      let result = b.collect::<Vec<_>>();
      if result.is_empty() {
        return Self::Recent::empty();
      } else {
        result
      }
    } else {
      return Self::Recent::empty();
    };

    // Aggregate the result using aggregator
    let result = self.agg.aggregate(batch, self.ctx);
    Self::Recent::singleton(result.into_iter())
  }
}

impl<'a, A, D, T1, T> Clone for AggregationSingleGroup<'a, A, D, T1, T>
where
  T1: StaticTupleTrait,
  D: Dataflow<T1, T>,
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
