use std::collections::*;

use itertools::iproduct;

use super::*;
use crate::utils::IdAllocator;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
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

impl std::fmt::Debug for Proof {
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

#[derive(Clone, PartialEq, Eq)]
pub struct Proofs {
  pub proofs: BTreeSet<Proof>,
}

impl Proofs {
  pub fn zero() -> Self {
    Self { proofs: BTreeSet::new() }
  }

  pub fn one() -> Self {
    Self {
      proofs: BTreeSet::from_iter(std::iter::once(Proof { facts: BTreeSet::new() })),
    }
  }

  pub fn singleton(f: usize) -> Self {
    let proof = Proof {
      facts: std::iter::once(f).collect(),
    };
    Self {
      proofs: BTreeSet::from_iter(std::iter::once(proof)),
    }
  }

  pub fn union(t1: &Self, t2: &Self) -> Self {
    Self {
      proofs: BTreeSet::from_iter(t1.proofs.iter().cloned().chain(t2.proofs.iter().cloned())),
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

impl std::fmt::Debug for Proofs {
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

impl Tag for Proofs {}

#[derive(Clone, Default)]
pub struct ProofsProvenance {
  id_allocator: IdAllocator,
  disjunctions: Disjunctions,
}

impl Provenance for ProofsProvenance {
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

  fn tagging_disjunction_optional_fn(&mut self, tags: Vec<Option<Self::InputTag>>) -> Vec<Self::Tag> {
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

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    t_old == t_new
  }
}
