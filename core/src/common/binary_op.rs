#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BinaryOp {
  Add,
  Sub,
  Mul,
  Div,
  Mod,
  And,
  Or,
  Xor,
  Eq,
  Neq,
  Lt,
  Leq,
  Gt,
  Geq,
}

impl std::fmt::Display for BinaryOp {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Add => f.write_str("+"),
      Self::Sub => f.write_str("-"),
      Self::Mul => f.write_str("*"),
      Self::Div => f.write_str("/"),
      Self::Mod => f.write_str("%"),
      Self::And => f.write_str("&&"),
      Self::Or => f.write_str("||"),
      Self::Xor => f.write_str("^"),
      Self::Eq => f.write_str("=="),
      Self::Neq => f.write_str("!="),
      Self::Lt => f.write_str("<"),
      Self::Leq => f.write_str("<="),
      Self::Gt => f.write_str(">"),
      Self::Geq => f.write_str(">="),
    }
  }
}

impl BinaryOp {
  pub fn is_arith(&self) -> bool {
    match self {
      Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Mod => true,
      _ => false,
    }
  }

  pub fn is_add_sub(&self) -> bool {
    match self {
      Self::Add | Self::Sub => true,
      _ => false,
    }
  }

  pub fn is_logical(&self) -> bool {
    match self {
      Self::And | Self::Or | Self::Xor => true,
      _ => false,
    }
  }

  pub fn is_eq_neq(&self) -> bool {
    match self {
      Self::Eq | Self::Neq => true,
      _ => false,
    }
  }

  pub fn is_eq(&self) -> bool {
    match self {
      Self::Eq => true,
      _ => false,
    }
  }

  pub fn is_numeric_cmp(&self) -> bool {
    match self {
      Self::Lt | Self::Leq | Self::Gt | Self::Geq => true,
      _ => false,
    }
  }

  pub fn add_sub_inv_op(&self) -> Option<Self> {
    match self {
      Self::Add => Some(Self::Sub),
      Self::Sub => Some(Self::Add),
      _ => None,
    }
  }
}
