use std::collections::*;

use itertools::iproduct;

use crate::utils::*;

use super::*;

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

  pub fn from_facts<I: Iterator<Item = usize>>(i: I) -> Self {
    Self {
      facts: BTreeSet::from_iter(i),
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
    Self {
      proofs: BTreeSet::new(),
    }
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

  pub fn from_proofs<I: Iterator<Item = Proof>>(i: I) -> Self {
    Self {
      proofs: BTreeSet::from_iter(i),
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

pub struct ProofsProvenance<P: PointerFamily> {
  id_allocator: P::Cell<IdAllocator>,
  disjunctions: P::Cell<Disjunctions>,
}

impl<P: PointerFamily> Default for ProofsProvenance<P> {
  fn default() -> Self {
    Self {
      id_allocator: P::new_cell(IdAllocator::default()),
      disjunctions: P::new_cell(Disjunctions::default()),
    }
  }
}

impl<P: PointerFamily> Clone for ProofsProvenance<P> {
  fn clone(&self) -> Self {
    Self {
      id_allocator: P::clone_cell(&self.id_allocator),
      disjunctions: P::clone_cell(&self.disjunctions),
    }
  }
}

impl<P: PointerFamily> Provenance for ProofsProvenance<P> {
  type Tag = Proofs;

  type InputTag = Exclusion;

  type OutputTag = Proofs;

  fn name() -> &'static str {
    "proofs"
  }

  fn tagging_fn(&self, exclusion: Self::InputTag) -> Self::Tag {
    let fact_id = P::get_cell_mut(&self.id_allocator, |a| a.alloc());

    // Disjunction id
    if let Exclusion::Exclusive(disjunction_id) = exclusion {
      P::get_cell_mut(&self.disjunctions, |d| d.add_disjunction(disjunction_id, fact_id));
    }

    // Return the proof
    Self::Tag::singleton(fact_id)
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
      .retain(|proof| !P::get_cell(&self.disjunctions, |d| d.has_conflict(&proof.facts)));
    prod
  }

  fn negate(&self, _: &Self::Tag) -> Option<Self::Tag> {
    panic!("Not implemented")
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    t_old == t_new
  }
}
