use super::as_boolean_formula::*;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct FactId(pub usize);

impl AsBooleanFormula for FactId {
  fn as_boolean_formula(&self) -> sdd::BooleanFormula {
    sdd::bf_pos(self.0)
  }
}
