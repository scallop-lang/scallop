use super::*;

/// A symbolic value of the tensor
#[derive(Debug, Clone, PartialEq, PartialOrd, Hash)]
pub struct TensorValue {
  pub shape: TensorShape,
  pub expr: TensorExpr,
}

impl TensorValue {
  pub fn is_scalar(&self) -> bool {
    self.shape.dim() == 0
  }

  pub fn add(self, v2: TensorValue) -> Option<TensorValue> {
    if self.shape == v2.shape {
      Some(TensorValue {
        shape: self.shape,
        expr: self.expr + v2.expr,
      })
    } else {
      None
    }
  }

  pub fn sub(self, v2: TensorValue) -> Option<TensorValue> {
    if self.shape == v2.shape {
      Some(TensorValue {
        shape: self.shape,
        expr: self.expr - v2.expr,
      })
    } else {
      None
    }
  }

  pub fn mul(self, v2: TensorValue) -> Option<TensorValue> {
    if self.shape == v2.shape || v2.is_scalar() {
      Some(TensorValue {
        shape: self.shape,
        expr: self.expr * v2.expr,
      })
    } else if v2.is_scalar() {
      Some(TensorValue {
        shape: v2.shape,
        expr: self.expr * v2.expr,
      })
    } else {
      None
    }
  }

  pub fn dot(self, v2: TensorValue) -> Option<TensorValue> {
    if self.shape.dim() == 1 && self.shape == v2.shape {
      Some(TensorValue {
        shape: TensorShape::scalar(),
        expr: self.expr.dot(v2.expr),
      })
    } else {
      None
    }
  }
}

impl From<f64> for TensorValue {
  fn from(value: f64) -> Self {
    Self {
      shape: TensorShape::scalar(),
      expr: TensorExpr::Float(value),
    }
  }
}

impl From<TensorSymbol> for TensorValue {
  fn from(value: TensorSymbol) -> Self {
    Self {
      shape: value.shape.clone(),
      expr: TensorExpr::Symbol(value),
    }
  }
}

impl std::fmt::Display for TensorValue {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.expr.fmt(f)
  }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum TensorExpr {
  /// A base tensor
  Symbol(TensorSymbol),

  /// Constant floating point
  Float(f64),

  /// A sum of two tensors
  Add(Box<TensorExpr>, Box<TensorExpr>),

  /// A subtraction between two tensors
  Sub(Box<TensorExpr>, Box<TensorExpr>),

  /// An element-wise multiplication of two tensors
  Mul(Box<TensorExpr>, Box<TensorExpr>),

  /// A dot product between two tensors
  Dot(Box<TensorExpr>, Box<TensorExpr>),
}

impl TensorExpr {
  pub fn dot(self, other: Self) -> Self {
    Self::Dot(Box::new(self), Box::new(other))
  }
}

impl std::fmt::Display for TensorExpr {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Symbol(s) => s.fmt(f),
      Self::Float(n) => n.fmt(f),
      Self::Add(v1, v2) => f.write_fmt(format_args!("{} + {}", v1, v2)),
      Self::Sub(v1, v2) => f.write_fmt(format_args!("{} - {}", v1, v2)),
      Self::Mul(v1, v2) => f.write_fmt(format_args!("{} * {}", v1, v2)),
      Self::Dot(v1, v2) => f.write_fmt(format_args!("dot({}, {})", v1, v2)),
    }
  }
}

impl std::hash::Hash for TensorExpr {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    match self {
      Self::Symbol(s) => s.hash(state),
      Self::Float(f) => i64::from_ne_bytes(f.to_ne_bytes()).hash(state),
      Self::Add(v1, v2) => {
        "add".hash(state);
        v1.hash(state);
        v2.hash(state);
      }
      Self::Sub(v1, v2) => {
        "sub".hash(state);
        v1.hash(state);
        v2.hash(state);
      }
      Self::Mul(v1, v2) => {
        "mul".hash(state);
        v1.hash(state);
        v2.hash(state);
      }
      Self::Dot(v1, v2) => {
        "dot".hash(state);
        v1.hash(state);
        v2.hash(state);
      }
    }
  }
}

impl From<TensorSymbol> for TensorExpr {
  fn from(sym: TensorSymbol) -> Self {
    Self::Symbol(sym)
  }
}

impl std::ops::Add<TensorExpr> for TensorExpr {
  type Output = TensorExpr;

  fn add(self, rhs: TensorExpr) -> Self::Output {
    TensorExpr::Add(Box::new(self), Box::new(rhs))
  }
}

impl std::ops::Sub<TensorExpr> for TensorExpr {
  type Output = TensorExpr;

  fn sub(self, rhs: TensorExpr) -> Self::Output {
    TensorExpr::Sub(Box::new(self), Box::new(rhs))
  }
}

impl std::ops::Mul<TensorExpr> for TensorExpr {
  type Output = TensorExpr;

  fn mul(self, rhs: TensorExpr) -> Self::Output {
    TensorExpr::Mul(Box::new(self), Box::new(rhs))
  }
}
