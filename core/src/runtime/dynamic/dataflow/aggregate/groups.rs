use std::collections::*;

use super::super::*;
use super::*;
use crate::common::tuple::Tuple;

#[derive(Clone)]
pub enum DynamicGroups<'a, T: Tag> {
  SingleCollection(&'a DynamicCollection<T>),
  GroupByKey(&'a DynamicCollection<T>),

  /// Group by joining on key
  ///
  /// The first collection has domain (U, V)
  /// The second collection has domain (U, W)
  ///
  /// The aggregation is performed on W, given a u \in U, returning values in A
  ///
  /// The output type of this aggregation is (U, V, A)
  GroupByJoinCollection(&'a DynamicCollection<T>, &'a DynamicCollection<T>),
}

impl<'a, T: Tag> DynamicGroups<'a, T> {
  pub fn from_collection(c: &'a DynamicCollection<T>) -> Self {
    Self::SingleCollection(c)
  }

  pub fn group_from_collection(c: &'a DynamicCollection<T>) -> Self {
    Self::GroupByKey(c)
  }

  pub fn iter_groups(&self) -> DynamicGroupsIterator<T> {
    match self {
      Self::SingleCollection(c) => DynamicGroupsIterator::single_collection(c),
      Self::GroupByKey(c) => {
        // Collect the key and the values
        let mut map = BTreeMap::<Tuple, Vec<DynamicElement<T>>>::new();
        for elem in &c.elements {
          map
            .entry(elem.tuple[0].clone())
            .or_default()
            .push(elem.clone());
        }

        // Create the groups
        let groups = map
          .into_iter()
          .map(|(key, mut elems)| {
            // First clean up the elements
            elems.sort();
            elems.dedup();

            // Then create the group
            DynamicGroup::KeyedBatch {
              key,
              batch: elems,
            }
          })
          .collect::<Vec<_>>();

        // Create the group iterator
        DynamicGroupsIterator::from_groups(groups)
      }
      Self::GroupByJoinCollection(group_by_c, main_c) => {
        let mut groups = vec![];

        // Collect keys by iterating through all the groups
        let (mut i, mut j) = (0, 0);
        while i < group_by_c.len() {
          let key_tag = &group_by_c[i].tag;
          let key_tup = &group_by_c[i].tuple;

          // If there is still an element
          if j < main_c.len() {
            let to_agg_tup = &main_c[j].tuple;

            // Compare the keys
            if key_tup[0] == to_agg_tup[0] {
              let key = key_tup[0].clone();

              // Get the set of variables to join on
              let mut to_join = vec![(key_tag.clone(), key_tup[1].clone())];
              let mut ip = i + 1;
              while ip < group_by_c.len() && group_by_c[ip].tuple[0] == key {
                let elem = &group_by_c[ip];
                to_join.push((elem.tag.clone(), elem.tuple[1].clone()));
                ip += 1;
              }

              // Move i forward to ip
              i = ip;

              // Get the set of elements to aggregate on
              let mut to_agg = vec![main_c[j].clone()];
              let mut jp = j + 1;
              while jp < main_c.len() && main_c[jp].tuple[0] == key {
                to_agg.push(main_c[jp].clone());
                jp += 1;
              }

              // Move j forward to jp
              j = jp;

              // Add this to the groups
              groups.push(DynamicGroup::JoinKeyBatch {
                key,
                to_join,
                batch: to_agg,
              });
            } else if key_tup[0] < to_agg_tup[0] {
              groups.push(DynamicGroup::JoinKeyBatch {
                key: key_tup[0].clone(),
                to_join: vec![(key_tag.clone(), key_tup[1].clone())],
                batch: vec![],
              });
              i += 1;
            } else {
              j += 1;
            }
          } else {
            // If there is no element, but we still have a group,
            // we create an empty batch for the group
            groups.push(DynamicGroup::JoinKeyBatch {
              key: key_tup[0].clone(),
              to_join: vec![(key_tag.clone(), key_tup[1].clone())],
              batch: vec![],
            });
            i += 1;
          }
        }

        // Return the collected set of groups
        DynamicGroupsIterator::from_groups(groups)
      }
    }
  }

  pub fn aggregate(
    self,
    aggregator: DynamicAggregateOp,
    ctx: &'a T::Context,
  ) -> DynamicDataflow<'a, T> {
    DynamicDataflow::Aggregate(DynamicAggregationDataflow {
      source: self,
      aggregator,
      ctx,
    })
  }
}

#[derive(Clone)]
pub enum DynamicGroupsIterator<T: Tag> {
  Empty,
  Single(Option<DynamicGroup<T>>),
  Groups(std::vec::IntoIter<DynamicGroup<T>>),
}

impl<T: Tag> DynamicGroupsIterator<T> {
  pub fn single_collection(col: &DynamicCollection<T>) -> Self {
    Self::Single(Some(DynamicGroup::Batch {
      batch: col.elements.clone(),
    }))
  }

  pub fn from_groups(groups: Vec<DynamicGroup<T>>) -> Self {
    Self::Groups(groups.into_iter())
  }
}

impl<'a, T: Tag> Iterator for DynamicGroupsIterator<T> {
  type Item = DynamicGroup<T>;

  fn next(&mut self) -> Option<Self::Item> {
    match self {
      Self::Empty => None,
      Self::Single(s) => s.take(),
      Self::Groups(m) => m.next(),
    }
  }
}
