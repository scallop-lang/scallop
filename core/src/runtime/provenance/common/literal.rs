use super::as_boolean_formula::*;

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub enum Literal {
  Pos(usize),
  Neg(usize),
}

impl Literal {
  pub fn fact_id(&self) -> usize {
    match self {
      Self::Pos(f) => *f,
      Self::Neg(f) => *f,
    }
  }

  pub fn sign(&self) -> bool {
    match self {
      Self::Pos(_) => true,
      Self::Neg(_) => false,
    }
  }

  pub fn negate(&self) -> Self {
    match self {
      Self::Pos(v) => Self::Neg(v.clone()),
      Self::Neg(v) => Self::Pos(v.clone()),
    }
  }
}

impl AsBooleanFormula for Literal {
  fn as_boolean_formula(&self) -> sdd::BooleanFormula {
    match self {
      Self::Pos(i) => sdd::bf_pos(*i),
      Self::Neg(i) => sdd::bf_neg(*i),
    }
  }
}

impl std::fmt::Display for Literal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Pos(i) => f.write_fmt(format_args!("pos({})", i)),
      Self::Neg(i) => f.write_fmt(format_args!("neg({})", i)),
    }
  }
}
