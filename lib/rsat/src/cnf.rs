/// Variable is a variable ID (potentially starting from 0)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Variable(usize);

impl Variable {
  pub fn new(var_id: usize) -> Self {
    Self(var_id)
  }

  pub fn variable_id(&self) -> usize {
    self.0
  }
}

/// Literal is either a positive or negative variable
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Literal(i64);

impl Default for Literal {
  fn default() -> Self {
    Self(0)
  }
}

impl std::fmt::Debug for Literal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.is_positive() {
      f.write_fmt(format_args!("Pos({})", self.variable_id()))
    } else {
      f.write_fmt(format_args!("Neg({})", self.variable_id()))
    }
  }
}

impl Literal {
  pub fn raw_id(&self) -> i64 {
    self.0
  }

  /// Create a positive literal from a variable
  pub fn positive(var: Variable) -> Self {
    Self(var.variable_id() as i64 + 1)
  }

  /// Create a negative literal from a variable
  pub fn negative(var: Variable) -> Self {
    Self(-(var.variable_id() as i64 + 1))
  }

  /// Get the variable of this literal
  pub fn variable(&self) -> Variable {
    Variable::new(self.variable_id())
  }

  /// Get the variable id in this literal
  pub fn variable_id(&self) -> usize {
    (self.0.abs() - 1) as usize
  }

  /// Check if the literal is positive
  pub fn is_positive(&self) -> bool {
    self.0 > 0
  }

  /// Check if the literal is negative
  pub fn is_negative(&self) -> bool {
    self.0 < 0
  }

  /// Get the negated literal
  pub fn negate(&self) -> Literal {
    Self(-self.0)
  }

  pub fn is_negate_of(&self, other: Literal) -> bool {
    self.0 == -other.0
  }
}

/// An out-facing data structure to represent a CNF (conjunctive normal form) formula.
pub type CNF = Vec<Vec<Literal>>;
