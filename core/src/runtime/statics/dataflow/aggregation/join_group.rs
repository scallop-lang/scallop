use std::marker::PhantomData;

use itertools::iproduct;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

use super::super::*;

pub struct AggregationJoinGroup<'a, A, D1, D2, K, T1, T2, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  D1: Dataflow<(K, T1), Prov>,
  D2: Dataflow<(K, T2), Prov>,
  A: Aggregator<T2, Prov>,
  Prov: Provenance,
{
  agg: A,
  d1: D1,
  d2: D2,
  ctx: &'a Prov,
  phantom: PhantomData<(K, T1, T2)>,
}

impl<'a, A, D1, D2, K, T1, T2, Prov> AggregationJoinGroup<'a, A, D1, D2, K, T1, T2, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  D1: Dataflow<(K, T1), Prov>,
  D2: Dataflow<(K, T2), Prov>,
  A: Aggregator<T2, Prov>,
  Prov: Provenance,
{
  pub fn new(agg: A, d1: D1, d2: D2, ctx: &'a Prov) -> Self {
    Self {
      agg,
      d1,
      d2,
      ctx,
      phantom: PhantomData,
    }
  }
}

impl<'a, A, D1, D2, K, T1, T2, Prov> Dataflow<(K, T1, A::Output), Prov>
  for AggregationJoinGroup<'a, A, D1, D2, K, T1, T2, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  D1: Dataflow<(K, T1), Prov>,
  D2: Dataflow<(K, T2), Prov>,
  A: Aggregator<T2, Prov>,
  Prov: Provenance,
{
  type Stable = EmptyBatches<std::iter::Empty<StaticElement<(K, T1, A::Output), Prov>>>;

  type Recent = SingleBatch<std::vec::IntoIter<StaticElement<(K, T1, A::Output), Prov>>>;

  fn iter_stable(&self) -> Self::Stable {
    Self::Stable::default()
  }

  fn iter_recent(self) -> Self::Recent {
    let mut group_by_c = if let Some(b) = self.d1.iter_recent().next() {
      b
    } else {
      return Self::Recent::empty();
    };

    let mut main_c = if let Some(b) = self.d2.iter_recent().next() {
      b
    } else {
      return Self::Recent::empty();
    };

    let agg = self.agg;
    let ctx = self.ctx;

    let mut groups = vec![];

    // Collect keys by iterating through all the groups
    let (mut i, mut j) = (group_by_c.next(), main_c.next());
    while let Some(group_by_elem) = &i {
      let key_tag = &group_by_elem.tag;
      let key_tup = &group_by_elem.tuple;

      // If there is still an element
      if let Some(to_agg_elem) = &j {
        let to_agg_tup = &to_agg_elem.tuple;

        // Compare the keys
        if key_tup.0 == to_agg_tup.0 {
          let key = key_tup.0.clone();

          // Get the set of variables to join on
          let mut to_join = vec![(key_tag.clone(), key_tup.1.clone())];
          i = group_by_c.next();
          while let Some(e) = &i {
            if e.tuple.0 == key {
              to_join.push((e.tag.clone(), e.tuple.1.clone()));
              i = group_by_c.next();
            } else {
              break;
            }
          }

          // Get the set of elements to aggregate on
          let mut to_agg = vec![to_agg_elem.clone()];
          j = main_c.next();
          while let Some(e) = &j {
            if e.tuple.0 == key {
              to_agg.push(e.clone());
              j = main_c.next();
            } else {
              break;
            }
          }

          // Add this to the groups
          groups.push((key, to_join, to_agg));
        } else if key_tup.0 < to_agg_tup.0 {
          groups.push((key_tup.0.clone(), vec![(key_tag.clone(), key_tup.1.clone())], vec![]));
          i = group_by_c.next();
        } else {
          j = main_c.next();
        }
      } else {
        // If there is no element, but we still have a group,
        // we create an empty batch for the group
        groups.push((key_tup.0.clone(), vec![(key_tag.clone(), key_tup.1.clone())], vec![]));
        i = group_by_c.next();
      }
    }

    let result: StaticElements<(K, T1, A::Output), Prov> = groups
      .into_iter()
      .map(|(group_key, group_by_vals, to_agg_vals)| {
        let to_agg_tups = to_agg_vals
          .iter()
          .map(|e| StaticElement::new(e.tuple.1.clone(), e.tag.clone()))
          .collect::<Vec<_>>();
        let agg_results = agg.aggregate(to_agg_tups, ctx);
        iproduct!(group_by_vals, agg_results)
          .map(|((tag, t1), agg_result)| {
            StaticElement::new(
              (group_key.clone(), t1.clone(), agg_result.tuple.get().clone()),
              ctx.mult(&tag, &agg_result.tag),
            )
          })
          .collect::<Vec<_>>()
      })
      .flatten()
      .collect::<Vec<_>>();

    Self::Recent::singleton(result.into_iter())
  }
}

impl<'a, A, D1, D2, K, T1, T2, Prov> Clone for AggregationJoinGroup<'a, A, D1, D2, K, T1, T2, Prov>
where
  K: StaticTupleTrait,
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  D1: Dataflow<(K, T1), Prov>,
  D2: Dataflow<(K, T2), Prov>,
  A: Aggregator<T2, Prov>,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self {
      agg: self.agg.clone(),
      d1: self.d1.clone(),
      d2: self.d2.clone(),
      ctx: self.ctx,
      phantom: PhantomData,
    }
  }
}
