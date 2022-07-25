use std::collections::BTreeSet;
use std::iter::FromIterator;

use scallop_core::runtime::provenance::*;

#[test]
fn test_disjunction_conflict_1() {
  let disj = Disjunction::from_iter(vec![1, 2, 3]);
  let facts = BTreeSet::from_iter(vec![2, 3, 4]);
  assert!(disj.has_conflict(&facts))
}

#[test]
fn test_disjunction_conflict_2() {
  let disj = Disjunction::from_iter(vec![1, 2, 3]);
  let facts = BTreeSet::from_iter(vec![3, 4, 5]);
  assert!(!disj.has_conflict(&facts))
}
