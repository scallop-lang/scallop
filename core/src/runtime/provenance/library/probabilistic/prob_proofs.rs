use std::collections::*;

use itertools::iproduct;

use super::disjunction::Disjunctions;
use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbProof {
  pub facts: BTreeSet<usize>,
}

impl ProbProof {
  pub fn merge(p1: &Self, p2: &Self) -> Self {
    Self {
      facts: p1.facts.iter().chain(p2.facts.iter()).cloned().collect(),
    }
  }
}

impl AsBooleanFormula for ProbProof {
  fn as_boolean_formula(&self) -> sdd::BooleanFormula {
    sdd::bf_conjunction(self.facts.iter().map(|f| sdd::bf_pos(f.clone())))
  }
}

impl std::fmt::Display for ProbProof {
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
pub struct ProbProofs {
  pub proofs: Vec<ProbProof>,
}

impl ProbProofs {
  pub fn zero() -> Self {
    Self { proofs: Vec::new() }
  }

  pub fn one() -> Self {
    Self {
      proofs: vec![ProbProof { facts: BTreeSet::new() }],
    }
  }

  pub fn singleton(f: usize) -> Self {
    Self {
      proofs: vec![ProbProof {
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
        .map(|(p1, p2)| ProbProof::merge(p1, p2))
        .collect(),
    }
  }
}

impl AsBooleanFormula for ProbProofs {
  fn as_boolean_formula(&self) -> sdd::BooleanFormula {
    sdd::bf_disjunction(self.proofs.iter().map(|p| p.as_boolean_formula()))
  }
}

impl std::fmt::Display for ProbProofs {
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

impl Tag for ProbProofs {
  type Context = ProbProofsContext;
}

#[derive(Clone, Default)]
pub struct ProbProofsContext {
  probabilities: Vec<f64>,
  disjunctions: Disjunctions,
}

impl ProvenanceContext for ProbProofsContext {
  type Tag = ProbProofs;

  type InputTag = f64;

  type OutputTag = f64;

  fn name() -> &'static str {
    "prob-proofs"
  }

  fn tagging_fn(&mut self, prob: Self::InputTag) -> Self::Tag {
    let id = self.probabilities.len();
    self.probabilities.push(prob);
    Self::Tag::singleton(id)
  }

  fn tagging_disjunction_fn(&mut self, tags: Vec<Self::InputTag>) -> Vec<Self::Tag> {
    // Add base disjunctions
    let ids = tags
      .into_iter()
      .map(|tag| {
        let id = self.probabilities.len();
        self.probabilities.push(tag);
        id
      })
      .collect::<Vec<_>>();

    // Add disjunction
    self.disjunctions.add_disjunction(ids.clone().into_iter());

    // Return tags
    ids.into_iter().map(Self::Tag::singleton).collect()
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    let s = RealSemiring;
    let v = |i: &usize| -> f64 { self.probabilities[*i] };
    AsBooleanFormula::wmc(t, &s, &v)
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
