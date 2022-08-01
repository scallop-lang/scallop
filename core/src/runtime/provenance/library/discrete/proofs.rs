use std::collections::*;

use itertools::iproduct;

use super::disjunction::Disjunctions;
use super::*;
use crate::utils::IdAllocator;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
  pub facts: BTreeSet<usize>,
}

impl Proof {
  pub fn merge(p1: &Self, p2: &Self) -> Self {
    Self {
      facts: p1.facts.iter().chain(p2.facts.iter()).cloned().collect(),
    }
  }
}

impl std::fmt::Display for Proof {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!(
      "{{{}}}",
      self
        .facts
        .iter()
        .map(|i| format!("{}", i))
        .collect::<Vec<_>>()
        .join(", ")
    ))
  }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proofs {
  pub proofs: Vec<Proof>,
}

impl Proofs {
  pub fn zero() -> Self {
    Self { proofs: Vec::new() }
  }

  pub fn one() -> Self {
    Self {
      proofs: vec![Proof { facts: BTreeSet::new() }],
    }
  }

  pub fn singleton(f: usize) -> Self {
    Self {
      proofs: vec![Proof {
        facts: std::iter::once(f).collect(),
      }],
    }
  }

  pub fn union(t1: &Self, t2: &Self) -> Self {
    Self {
      proofs: vec![t1.proofs.clone(), t2.proofs.clone()].concat(),
    }
  }

  pub fn cartesian_product(t1: &Self, t2: &Self) -> Self {
    Self {
      proofs: iproduct!(&t1.proofs, &t2.proofs)
        .into_iter()
        .map(|(p1, p2)| Proof::merge(p1, p2))
        .collect(),
    }
  }
}

impl std::fmt::Display for Proofs {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!(
      "{{{}}}",
      self
        .proofs
        .iter()
        .map(|p| format!("{}", p))
        .collect::<Vec<_>>()
        .join(", ")
    ))
  }
}

impl Tag for Proofs {
  type Context = ProofsContext;
}

#[derive(Clone, Default)]
pub struct ProofsContext {
  id_allocator: IdAllocator,
  disjunctions: Disjunctions,
}

impl ProvenanceContext for ProofsContext {
  type Tag = Proofs;

  type InputTag = ();

  type OutputTag = Proofs;

  fn name() -> &'static str {
    "proofs"
  }

  fn tagging_fn(&mut self, _: Self::InputTag) -> Self::Tag {
    let id = self.id_allocator.alloc();
    Self::Tag::singleton(id)
  }

  fn tagging_disjunction_fn(&mut self, tags: Vec<Self::InputTag>) -> Vec<Self::Tag> {
    let ids = tags.into_iter().map(|_| self.id_allocator.alloc()).collect::<Vec<_>>();
    self.disjunctions.add_disjunction(ids.clone().into_iter());
    ids.into_iter().map(Self::Tag::singleton).collect()
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    t.clone()
  }

  fn discard(&self, t: &Self::Tag) -> bool {
    t.proofs.is_empty()
  }

  fn zero(&self) -> Self::Tag {
    Self::Tag::zero()
  }

  fn one(&self) -> Self::Tag {
    Self::Tag::one()
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    Self::Tag::union(t1, t2)
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    let mut prod = Self::Tag::cartesian_product(t1, t2);
    prod
      .proofs
      .retain(|proof| !self.disjunctions.has_conflict(&proof.facts));
    prod
  }

  fn negate(&self, _: &Self::Tag) -> Option<Self::Tag> {
    panic!("Not implemented")
  }

  fn minus(&self, _: &Self::Tag, _: &Self::Tag) -> Option<Self::Tag> {
    panic!("Not implemented")
  }
}
