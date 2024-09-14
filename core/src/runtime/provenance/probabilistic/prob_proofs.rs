use std::collections::*;

use itertools::iproduct;

use crate::utils::*;

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

impl Tag for ProbProofs {}

#[derive(Default)]
pub struct ProbProofsProvenance<P: PointerFamily = RcFamily> {
  probs: P::Cell<Vec<f64>>,
  disjunctions: P::Cell<Disjunctions>,
}

impl<P: PointerFamily> Clone for ProbProofsProvenance<P> {
  fn clone(&self) -> Self {
    Self {
      probs: P::clone_cell(&self.probs),
      disjunctions: P::clone_cell(&self.disjunctions),
    }
  }
}

impl<P: PointerFamily> ProbProofsProvenance<P> {
  fn fact_probability(&self, i: &usize) -> f64 {
    P::get_cell(&self.probs, |p| p[*i])
  }
}

impl<P: PointerFamily> Provenance for ProbProofsProvenance<P> {
  type Tag = ProbProofs;

  type InputTag = InputExclusiveProb;

  type OutputTag = f64;

  fn name(&self) -> String {
    format!("prob-proofs")
  }

  fn tagging_fn(&self, input_tag: Self::InputTag) -> Self::Tag {
    // First generate id and push the probability into the list
    let fact_id = P::get_cell(&self.probs, |p| p.len());
    P::get_cell_mut(&self.probs, |p| p.push(input_tag.prob));

    // Add exlusion if needed
    if let Some(disj_id) = input_tag.exclusion {
      P::get_cell_mut(&self.disjunctions, |d| d.add_disjunction(disj_id, fact_id));
    }

    // Lastly return a tag
    Self::Tag::singleton(fact_id)
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    let s = RealSemiring;
    let v = |i: &usize| -> f64 { self.fact_probability(i) };
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
      .retain(|proof| P::get_cell(&self.disjunctions, |d| !d.has_conflict(&proof.facts)));
    prod
  }

  fn negate(&self, _: &Self::Tag) -> Option<Self::Tag> {
    unimplemented!()
  }

  fn minus(&self, _: &Self::Tag, _: &Self::Tag) -> Option<Self::Tag> {
    unimplemented!()
  }

  fn saturated(&self, _: &Self::Tag, _: &Self::Tag) -> bool {
    unimplemented!()
  }

  fn weight(&self, t: &Self::Tag) -> f64 {
    let s = RealSemiring;
    let v = |i: &usize| -> f64 { self.fact_probability(i) };
    AsBooleanFormula::wmc(t, &s, &v)
  }
}
