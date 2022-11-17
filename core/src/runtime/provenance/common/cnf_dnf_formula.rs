use super::{AsBooleanFormula, Clause, Literal};
use super::super::*;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum FormulaKind {
  CNF,
  DNF,
}

impl FormulaKind {
  pub fn name(&self) -> &'static str {
    match self {
      Self::CNF => "CNF",
      Self::DNF => "DNF",
    }
  }

  pub fn is_cnf(&self) -> bool {
    match self {
      Self::CNF => true,
      Self::DNF => false,
    }
  }

  pub fn is_dnf(&self) -> bool {
    match self {
      Self::CNF => false,
      Self::DNF => true,
    }
  }

  pub fn counterpart(&self) -> Self {
    match self {
      Self::CNF => Self::DNF,
      Self::DNF => Self::CNF,
    }
  }
}

/// A set of proofs that could represent either a CNF or DNF
#[derive(Clone, PartialEq, PartialOrd)]
pub struct CNFDNFFormula {
  pub kind: FormulaKind,
  pub clauses: Vec<Clause>,
}

impl std::fmt::Debug for CNFDNFFormula {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(self.kind.name())?;
    f.debug_set().entries(&self.clauses).finish()
  }
}

impl std::fmt::Display for CNFDNFFormula {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(self.kind.name())?;
    f.write_str("{")?;
    for (i, clause) in self.clauses.iter().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      f.write_fmt(format_args!("{}", clause))?;
    }
    f.write_str("}")
  }
}

impl CNFDNFFormula {
  pub fn new(kind: FormulaKind, clauses: Vec<Clause>) -> Self {
    Self { kind, clauses }
  }

  pub fn iter(&self) -> impl Iterator<Item = &Clause> + '_ {
    self.clauses.iter()
  }

  pub fn is_empty(&self) -> bool {
    self.clauses.is_empty()
  }

  pub fn cnf(clauses: Vec<Clause>) -> Self {
    Self {
      kind: FormulaKind::CNF,
      clauses,
    }
  }

  pub fn cnf_zero() -> Self {
    Self::cnf(vec![Clause::empty()])
  }

  pub fn cnf_one() -> Self {
    Self::cnf(vec![])
  }

  pub fn cnf_singleton(v: usize) -> Self {
    Self::cnf(vec![Clause::singleton(Literal::Pos(v))])
  }

  pub fn dnf(clauses: Vec<Clause>) -> Self {
    Self {
      kind: FormulaKind::DNF,
      clauses,
    }
  }

  pub fn dnf_zero() -> Self {
    Self::dnf(vec![])
  }

  pub fn dnf_one() -> Self {
    Self::dnf(vec![Clause::empty()])
  }

  pub fn dnf_singleton(v: usize) -> Self {
    Self::dnf(vec![Clause::singleton(Literal::Pos(v))])
  }

  pub fn is_zero(&self) -> bool {
    match &self.kind {
      FormulaKind::CNF => self.clauses.iter().any(|c| c.is_empty()),
      FormulaKind::DNF => self.clauses.is_empty(),
    }
  }

  pub fn negate(&self) -> Self {
    if self.clauses.len() == 1 && self.clauses[0].len() == 1 {
      Self::new(
        self.kind.clone(),
        vec![Clause::singleton(self.clauses[0].literals[0].negate())],
      )
    } else {
      let negate_clauses = self
        .clauses
        .iter()
        .map(|c| Clause::new(c.literals.iter().map(|l| l.negate()).collect()))
        .collect();
      Self::new(self.kind.counterpart(), negate_clauses)
    }
  }
}

impl AsBooleanFormula for CNFDNFFormula {
  fn as_boolean_formula(&self) -> sdd::BooleanFormula {
    match &self.kind {
      FormulaKind::CNF => sdd::bf_conjunction(
        self
          .clauses
          .iter()
          .map(|c| sdd::bf_disjunction(c.literals.iter().map(|l| l.as_boolean_formula()))),
      ),
      FormulaKind::DNF => sdd::bf_disjunction(
        self
          .clauses
          .iter()
          .map(|c| sdd::bf_conjunction(c.literals.iter().map(|l| l.as_boolean_formula()))),
      ),
    }
  }
}

impl Tag for CNFDNFFormula {}
