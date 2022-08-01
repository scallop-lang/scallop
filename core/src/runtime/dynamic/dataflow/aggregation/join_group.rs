use itertools::*;

use super::*;

pub struct DynamicAggregationJoinGroupDataflow<'a, T: Tag> {
  pub agg: DynamicAggregator,
  pub d1: Box<DynamicDataflow<'a, T>>,
  pub d2: Box<DynamicDataflow<'a, T>>,
  pub ctx: &'a T::Context,
}

impl<'a, T: Tag> Clone for DynamicAggregationJoinGroupDataflow<'a, T> {
  fn clone(&self) -> Self {
    Self {
      agg: self.agg.clone(),
      d1: self.d1.clone(),
      d2: self.d2.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, T: Tag> DynamicAggregationJoinGroupDataflow<'a, T> {
  pub fn new(
    agg: DynamicAggregator,
    d1: DynamicDataflow<'a, T>,
    d2: DynamicDataflow<'a, T>,
    ctx: &'a T::Context,
  ) -> Self {
    Self {
      agg,
      d1: Box::new(d1),
      d2: Box::new(d2),
      ctx,
    }
  }

  pub fn iter_stable(&self) -> DynamicBatches<'a, T> {
    DynamicBatches::empty()
  }

  pub fn iter_recent(&self) -> DynamicBatches<'a, T> {
    let mut group_by_c = if let Some(b1) = self.d1.iter_recent().next() {
      b1
    } else {
      return DynamicBatches::empty();
    };

    let mut main_c = if let Some(b2) = self.d2.iter_recent().next() {
      b2
    } else {
      return DynamicBatches::empty();
    };

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
        if key_tup[0] == to_agg_tup[0] {
          let key = key_tup[0].clone();

          // Get the set of variables to join on
          let mut to_join = vec![(key_tag.clone(), key_tup[1].clone())];
          i = group_by_c.next();
          while let Some(e) = &i {
            if e.tuple[0] == key {
              to_join.push((e.tag.clone(), e.tuple[1].clone()));
              i = group_by_c.next();
            } else {
              break;
            }
          }

          // Get the set of elements to aggregate on
          let mut to_agg = vec![to_agg_elem.clone()];
          j = main_c.next();
          while let Some(e) = &j {
            if e.tuple[0] == key {
              to_agg.push(e.clone());
              j = main_c.next();
            } else {
              break;
            }
          }

          // Add this to the groups
          groups.push((key, to_join, to_agg));
        } else if key_tup[0] < to_agg_tup[0] {
          groups.push((key_tup[0].clone(), vec![(key_tag.clone(), key_tup[1].clone())], vec![]));
          i = group_by_c.next();
        } else {
          j = main_c.next();
        }
      } else {
        // If there is no element, but we still have a group,
        // we create an empty batch for the group
        groups.push((key_tup[0].clone(), vec![(key_tag.clone(), key_tup[1].clone())], vec![]));
        i = group_by_c.next();
      }
    }

    let result: DynamicElements<T> = groups
      .into_iter()
      .map(|(group_key, group_by_vals, to_agg_vals)| {
        let to_agg_tups = to_agg_vals
          .iter()
          .map(|e| DynamicElement::new(e.tuple[1].clone(), e.tag.clone()))
          .collect::<Vec<_>>();
        let agg_results = self.agg.aggregate(to_agg_tups, self.ctx);
        iproduct!(group_by_vals, agg_results)
          .map(|((tag, t1), agg_result)| {
            DynamicElement::new(
              (group_key.clone(), t1.clone(), agg_result.tuple.clone()),
              self.ctx.mult(&tag, &agg_result.tag),
            )
          })
          .collect::<Vec<_>>()
      })
      .flatten()
      .collect::<Vec<_>>();

    DynamicBatches::single(DynamicBatch::source_vec(result))
  }
}
