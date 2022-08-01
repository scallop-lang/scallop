use super::binary_op::BinaryOp;
use super::tuple::Tuple;
use super::tuple_access::TupleAccessor;
use super::unary_op::UnaryOp;
use super::value::Value;
use super::value_type::ValueType;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Expr {
  Tuple(Vec<Expr>),
  Access(TupleAccessor),
  Constant(Value),
  Binary(BinaryExpr),
  Unary(UnaryExpr),
  IfThenElse(IfThenElseExpr),
}

impl Expr {
  pub fn binary(op: BinaryOp, op1: Expr, op2: Expr) -> Self {
    Self::Binary(BinaryExpr::new(op, op1, op2))
  }

  pub fn unary(op: UnaryOp, op1: Expr) -> Self {
    Self::Unary(UnaryExpr::new(op, op1))
  }

  pub fn access<T: Into<TupleAccessor>>(t: T) -> Self {
    Self::Access(t.into())
  }

  pub fn constant<T: Into<Value>>(t: T) -> Self {
    Self::Constant(t.into())
  }

  pub fn lt(self, other: Expr) -> Self {
    Self::Binary(BinaryExpr {
      op: BinaryOp::Lt,
      op1: Box::new(self),
      op2: Box::new(other),
    })
  }

  pub fn leq(self, other: Expr) -> Self {
    Self::Binary(BinaryExpr {
      op: BinaryOp::Leq,
      op1: Box::new(self),
      op2: Box::new(other),
    })
  }

  pub fn gt(self, other: Expr) -> Self {
    Self::Binary(BinaryExpr {
      op: BinaryOp::Gt,
      op1: Box::new(self),
      op2: Box::new(other),
    })
  }

  pub fn geq(self, other: Expr) -> Self {
    Self::Binary(BinaryExpr {
      op: BinaryOp::Geq,
      op1: Box::new(self),
      op2: Box::new(other),
    })
  }

  pub fn eval(&self, v: &Tuple) -> Tuple {
    match self {
      Self::Tuple(t) => Tuple::Tuple(t.iter().map(|e| e.eval(v)).collect()),
      Self::Access(a) => v[a].clone(),
      Self::Constant(c) => Tuple::Value(c.clone()),
      Self::Binary(b) => b.eval(v),
      Self::Unary(u) => u.eval(v),
      Self::IfThenElse(i) => i.eval(v),
    }
  }
}

impl std::ops::Add<Expr> for Expr {
  type Output = Self;

  fn add(self, other: Self) -> Self {
    Self::Binary(BinaryExpr {
      op: BinaryOp::Add,
      op1: Box::new(self),
      op2: Box::new(other),
    })
  }
}

impl std::ops::Sub<Expr> for Expr {
  type Output = Self;

  fn sub(self, other: Self) -> Self {
    Self::Binary(BinaryExpr {
      op: BinaryOp::Sub,
      op1: Box::new(self),
      op2: Box::new(other),
    })
  }
}

impl std::ops::Mul<Expr> for Expr {
  type Output = Self;

  fn mul(self, other: Self) -> Self {
    Self::Binary(BinaryExpr {
      op: BinaryOp::Mul,
      op1: Box::new(self),
      op2: Box::new(other),
    })
  }
}

impl std::ops::Div<Expr> for Expr {
  type Output = Self;

  fn div(self, other: Self) -> Self {
    Self::Binary(BinaryExpr {
      op: BinaryOp::Div,
      op1: Box::new(self),
      op2: Box::new(other),
    })
  }
}

impl std::ops::BitAnd<Expr> for Expr {
  type Output = Self;

  fn bitand(self, other: Self) -> Self {
    Self::Binary(BinaryExpr {
      op: BinaryOp::And,
      op1: Box::new(self),
      op2: Box::new(other),
    })
  }
}

impl std::ops::BitOr<Expr> for Expr {
  type Output = Self;

  fn bitor(self, other: Self) -> Self {
    Self::Binary(BinaryExpr {
      op: BinaryOp::Or,
      op1: Box::new(self),
      op2: Box::new(other),
    })
  }
}

impl std::ops::BitXor<Expr> for Expr {
  type Output = Self;

  fn bitxor(self, other: Self) -> Self {
    Self::Binary(BinaryExpr {
      op: BinaryOp::Xor,
      op1: Box::new(self),
      op2: Box::new(other),
    })
  }
}

impl From<()> for Expr {
  fn from((): ()) -> Self {
    Expr::Tuple(vec![])
  }
}

impl<A> From<(A,)> for Expr
where
  A: Into<Expr>,
{
  fn from((a,): (A,)) -> Self {
    Expr::Tuple(vec![a.into()])
  }
}

impl<A, B> From<(A, B)> for Expr
where
  A: Into<Expr>,
  B: Into<Expr>,
{
  fn from((a, b): (A, B)) -> Self {
    Expr::Tuple(vec![a.into(), b.into()])
  }
}

impl<A, B, C> From<(A, B, C)> for Expr
where
  A: Into<Expr>,
  B: Into<Expr>,
  C: Into<Expr>,
{
  fn from((a, b, c): (A, B, C)) -> Self {
    Expr::Tuple(vec![a.into(), b.into(), c.into()])
  }
}

impl<A, B, C, D> From<(A, B, C, D)> for Expr
where
  A: Into<Expr>,
  B: Into<Expr>,
  C: Into<Expr>,
  D: Into<Expr>,
{
  fn from((a, b, c, d): (A, B, C, D)) -> Self {
    Expr::Tuple(vec![a.into(), b.into(), c.into(), d.into()])
  }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BinaryExpr {
  pub op: BinaryOp,
  pub op1: Box<Expr>,
  pub op2: Box<Expr>,
}

impl BinaryExpr {
  pub fn new(op: BinaryOp, op1: Expr, op2: Expr) -> Self {
    Self {
      op,
      op1: Box::new(op1),
      op2: Box::new(op2),
    }
  }

  pub fn eval(&self, v: &Tuple) -> Tuple {
    let lhs_v = self.op1.eval(v);
    let rhs_v = self.op2.eval(v);
    use crate::common::value::Value::*;
    use BinaryOp::*;
    match (&self.op, lhs_v, rhs_v) {
      // Addition
      (Add, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(I8(i1 + i2)),
      (Add, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(I16(i1 + i2)),
      (Add, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(I32(i1 + i2)),
      (Add, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(I64(i1 + i2)),
      (Add, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(I128(i1 + i2)),
      (Add, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(ISize(i1 + i2)),
      (Add, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(U8(i1 + i2)),
      (Add, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(U16(i1 + i2)),
      (Add, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(U32(i1 + i2)),
      (Add, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(U64(i1 + i2)),
      (Add, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(U128(i1 + i2)),
      (Add, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(USize(i1 + i2)),
      (Add, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(F32(i1 + i2)),
      (Add, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(F64(i1 + i2)),
      (Add, b1, b2) => panic!("Cannot perform ADD on {:?} and {:?}", b1, b2),

      // Subtraction
      (Sub, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(I8(i1 - i2)),
      (Sub, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(I16(i1 - i2)),
      (Sub, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(I32(i1 - i2)),
      (Sub, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(I64(i1 - i2)),
      (Sub, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(I128(i1 - i2)),
      (Sub, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(ISize(i1 - i2)),
      (Sub, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(U8(i1 - i2)),
      (Sub, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(U16(i1 - i2)),
      (Sub, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(U32(i1 - i2)),
      (Sub, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(U64(i1 - i2)),
      (Sub, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(U128(i1 - i2)),
      (Sub, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(USize(i1 - i2)),
      (Sub, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(F32(i1 - i2)),
      (Sub, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(F64(i1 - i2)),
      (Sub, b1, b2) => panic!("Cannot perform SUB on {:?} and {:?}", b1, b2),

      // Multiplication
      (Mul, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(I8(i1 * i2)),
      (Mul, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(I16(i1 * i2)),
      (Mul, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(I32(i1 * i2)),
      (Mul, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(I64(i1 * i2)),
      (Mul, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(I128(i1 * i2)),
      (Mul, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(ISize(i1 * i2)),
      (Mul, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(U8(i1 * i2)),
      (Mul, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(U16(i1 * i2)),
      (Mul, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(U32(i1 * i2)),
      (Mul, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(U64(i1 * i2)),
      (Mul, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(U128(i1 * i2)),
      (Mul, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(USize(i1 * i2)),
      (Mul, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(F32(i1 * i2)),
      (Mul, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(F64(i1 * i2)),
      (Mul, b1, b2) => panic!("Cannot perform MUL on {:?} and {:?}", b1, b2),

      // Division
      (Div, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(I8(i1 / i2)),
      (Div, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(I16(i1 / i2)),
      (Div, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(I32(i1 / i2)),
      (Div, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(I64(i1 / i2)),
      (Div, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(I128(i1 / i2)),
      (Div, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(ISize(i1 / i2)),
      (Div, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(U8(i1 / i2)),
      (Div, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(U16(i1 / i2)),
      (Div, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(U32(i1 / i2)),
      (Div, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(U64(i1 / i2)),
      (Div, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(U128(i1 / i2)),
      (Div, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(USize(i1 / i2)),
      (Div, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(F32(i1 / i2)),
      (Div, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(F64(i1 / i2)),
      (Div, b1, b2) => panic!("Cannot perform DIV on {:?} and {:?}", b1, b2),

      // Mod
      (Mod, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(I8(i1 % i2)),
      (Mod, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(I16(i1 % i2)),
      (Mod, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(I32(i1 % i2)),
      (Mod, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(I64(i1 % i2)),
      (Mod, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(I128(i1 % i2)),
      (Mod, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(ISize(i1 % i2)),
      (Mod, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(U8(i1 % i2)),
      (Mod, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(U16(i1 % i2)),
      (Mod, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(U32(i1 % i2)),
      (Mod, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(U64(i1 % i2)),
      (Mod, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(U128(i1 % i2)),
      (Mod, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(USize(i1 % i2)),
      (Mod, b1, b2) => panic!("Cannot perform MOD on {:?} and {:?}", b1, b2),

      // Boolean
      (And, Tuple::Value(Bool(b1)), Tuple::Value(Bool(b2))) => Tuple::Value(Bool(b1 && b2)),
      (And, b1, b2) => panic!("Cannot perform AND on {:?} and {:?}", b1, b2),
      (Or, Tuple::Value(Bool(b1)), Tuple::Value(Bool(b2))) => Tuple::Value(Bool(b1 || b2)),
      (Or, b1, b2) => panic!("Cannot perform OR on {:?} and {:?}", b1, b2),
      (Xor, Tuple::Value(Bool(b1)), Tuple::Value(Bool(b2))) => Tuple::Value(Bool(b1 ^ b2)),
      (Xor, b1, b2) => panic!("Cannot perform XOR on {:?} and {:?}", b1, b2),

      // Equal to
      (Eq, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(Char(i1)), Tuple::Value(Char(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(Bool(i1)), Tuple::Value(Bool(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(Str(i1)), Tuple::Value(Str(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, Tuple::Value(String(i1)), Tuple::Value(String(i2))) => Tuple::Value(Bool(i1 == i2)),
      // (Eq, Tuple::Value(RcString(i1)), Tuple::Value(RcString(i2))) => Tuple::Value(Bool(i1 == i2)),
      (Eq, b1, b2) => panic!("Cannot perform EQ on {:?} and {:?}", b1, b2),

      // Not equal to
      (Neq, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(Char(i1)), Tuple::Value(Char(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(Bool(i1)), Tuple::Value(Bool(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(Str(i1)), Tuple::Value(Str(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, Tuple::Value(String(i1)), Tuple::Value(String(i2))) => Tuple::Value(Bool(i1 != i2)),
      // (Neq, Tuple::Value(RcString(i1)), Tuple::Value(RcString(i2))) => Tuple::Value(Bool(i1 != i2)),
      (Neq, b1, b2) => panic!("Cannot perform NEQ on {:?} and {:?}", b1, b2),

      // Greater than
      (Gt, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(Bool(i1 > i2)),
      (Gt, b1, b2) => panic!("Cannot perform GT on {:?} and {:?}", b1, b2),

      // Greater than or equal to
      (Geq, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(Bool(i1 >= i2)),
      (Geq, b1, b2) => panic!("Cannot perform GEQ on {:?} and {:?}", b1, b2),

      // Less than
      (Lt, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(Bool(i1 < i2)),
      (Lt, b1, b2) => panic!("Cannot perform LT on {:?} and {:?}", b1, b2),

      // Less than or equal to
      (Leq, Tuple::Value(I8(i1)), Tuple::Value(I8(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(I16(i1)), Tuple::Value(I16(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(I32(i1)), Tuple::Value(I32(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(I64(i1)), Tuple::Value(I64(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(I128(i1)), Tuple::Value(I128(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(ISize(i1)), Tuple::Value(ISize(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(U8(i1)), Tuple::Value(U8(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(U16(i1)), Tuple::Value(U16(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(U32(i1)), Tuple::Value(U32(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(U64(i1)), Tuple::Value(U64(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(U128(i1)), Tuple::Value(U128(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(USize(i1)), Tuple::Value(USize(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(F32(i1)), Tuple::Value(F32(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, Tuple::Value(F64(i1)), Tuple::Value(F64(i2))) => Tuple::Value(Bool(i1 <= i2)),
      (Leq, b1, b2) => panic!("Cannot perform LEQ on {:?} and {:?}", b1, b2),
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct UnaryExpr {
  pub op: UnaryOp,
  pub op1: Box<Expr>,
}

impl UnaryExpr {
  pub fn new(op: UnaryOp, op1: Expr) -> Self {
    Self { op, op1: Box::new(op1) }
  }

  pub fn eval(&self, v: &Tuple) -> Tuple {
    let arg_v = self.op1.eval(v);
    use crate::common::value::Value::*;
    use UnaryOp::*;
    match (&self.op, arg_v) {
      // Negative
      (Neg, Tuple::Value(I8(i))) => Tuple::Value(I8(-i)),
      (Neg, Tuple::Value(I16(i))) => Tuple::Value(I16(-i)),
      (Neg, Tuple::Value(I32(i))) => Tuple::Value(I32(-i)),
      (Neg, Tuple::Value(I64(i))) => Tuple::Value(I64(-i)),
      (Neg, Tuple::Value(I128(i))) => Tuple::Value(I128(-i)),
      (Neg, Tuple::Value(ISize(i))) => Tuple::Value(ISize(-i)),
      (Neg, Tuple::Value(F32(i))) => Tuple::Value(F32(-i)),
      (Neg, Tuple::Value(F64(i))) => Tuple::Value(F64(-i)),
      (Neg, v) => panic!("Negate operation cannot be operating on value of type {:?}", v),

      // Positive
      (Pos, x) => x,

      // Not
      (Not, Tuple::Value(Bool(b))) => Tuple::Value(Bool(!b)),
      (Not, v) => panic!("Not operation cannot be operating on value of type {:?}", v),

      // Type cast
      (TypeCast(dst), arg) => {
        use ValueType as T;
        match (arg, dst) {
          (Tuple::Value(I8(i)), T::I8) => Tuple::Value(I8(i)),
          (Tuple::Value(I8(i)), T::I16) => Tuple::Value(I16(i as i16)),
          (Tuple::Value(I8(i)), T::I32) => Tuple::Value(I32(i as i32)),
          (Tuple::Value(I8(i)), T::I64) => Tuple::Value(I64(i as i64)),
          (Tuple::Value(I8(i)), T::I128) => Tuple::Value(I128(i as i128)),
          (Tuple::Value(I8(i)), T::ISize) => Tuple::Value(ISize(i as isize)),
          (Tuple::Value(I8(i)), T::U8) => Tuple::Value(U8(i as u8)),
          (Tuple::Value(I8(i)), T::U16) => Tuple::Value(U16(i as u16)),
          (Tuple::Value(I8(i)), T::U32) => Tuple::Value(U32(i as u32)),
          (Tuple::Value(I8(i)), T::U64) => Tuple::Value(U64(i as u64)),
          (Tuple::Value(I8(i)), T::U128) => Tuple::Value(U128(i as u128)),
          (Tuple::Value(I8(i)), T::USize) => Tuple::Value(USize(i as usize)),
          (Tuple::Value(I8(i)), T::F32) => Tuple::Value(F32(i as f32)),
          (Tuple::Value(I8(i)), T::F64) => Tuple::Value(F64(i as f64)),

          (Tuple::Value(I32(i)), T::I8) => Tuple::Value(I8(i as i8)),
          (Tuple::Value(I32(i)), T::I16) => Tuple::Value(I16(i as i16)),
          (Tuple::Value(I32(i)), T::I32) => Tuple::Value(I32(i)),
          (Tuple::Value(I32(i)), T::I64) => Tuple::Value(I64(i as i64)),
          (Tuple::Value(I32(i)), T::I128) => Tuple::Value(I128(i as i128)),
          (Tuple::Value(I32(i)), T::ISize) => Tuple::Value(ISize(i as isize)),
          (Tuple::Value(I32(i)), T::U8) => Tuple::Value(U8(i as u8)),
          (Tuple::Value(I32(i)), T::U16) => Tuple::Value(U16(i as u16)),
          (Tuple::Value(I32(i)), T::U32) => Tuple::Value(U32(i as u32)),
          (Tuple::Value(I32(i)), T::U64) => Tuple::Value(U64(i as u64)),
          (Tuple::Value(I32(i)), T::U128) => Tuple::Value(U128(i as u128)),
          (Tuple::Value(I32(i)), T::USize) => Tuple::Value(USize(i as usize)),
          (Tuple::Value(I32(i)), T::F32) => Tuple::Value(F32(i as f32)),
          (Tuple::Value(I32(i)), T::F64) => Tuple::Value(F64(i as f64)),

          (Tuple::Value(USize(i)), T::I8) => Tuple::Value(I8(i as i8)),
          (Tuple::Value(USize(i)), T::I16) => Tuple::Value(I16(i as i16)),
          (Tuple::Value(USize(i)), T::I32) => Tuple::Value(I32(i as i32)),
          (Tuple::Value(USize(i)), T::I64) => Tuple::Value(I64(i as i64)),
          (Tuple::Value(USize(i)), T::I128) => Tuple::Value(I128(i as i128)),
          (Tuple::Value(USize(i)), T::ISize) => Tuple::Value(ISize(i as isize)),
          (Tuple::Value(USize(i)), T::U8) => Tuple::Value(U8(i as u8)),
          (Tuple::Value(USize(i)), T::U16) => Tuple::Value(U16(i as u16)),
          (Tuple::Value(USize(i)), T::U32) => Tuple::Value(U32(i as u32)),
          (Tuple::Value(USize(i)), T::U64) => Tuple::Value(U64(i as u64)),
          (Tuple::Value(USize(i)), T::U128) => Tuple::Value(U128(i as u128)),
          (Tuple::Value(USize(i)), T::USize) => Tuple::Value(USize(i)),
          (Tuple::Value(USize(i)), T::F32) => Tuple::Value(F32(i as f32)),
          (Tuple::Value(USize(i)), T::F64) => Tuple::Value(F64(i as f64)),

          (Tuple::Value(I8(i)), T::String) => Tuple::Value(String(i.to_string())),
          (Tuple::Value(I16(i)), T::String) => Tuple::Value(String(i.to_string())),
          (Tuple::Value(I32(i)), T::String) => Tuple::Value(String(i.to_string())),
          (Tuple::Value(I64(i)), T::String) => Tuple::Value(String(i.to_string())),
          (Tuple::Value(I128(i)), T::String) => Tuple::Value(String(i.to_string())),
          (Tuple::Value(ISize(i)), T::String) => Tuple::Value(String(i.to_string())),
          (Tuple::Value(U8(u)), T::String) => Tuple::Value(String(u.to_string())),
          (Tuple::Value(U16(u)), T::String) => Tuple::Value(String(u.to_string())),
          (Tuple::Value(U32(u)), T::String) => Tuple::Value(String(u.to_string())),
          (Tuple::Value(U64(u)), T::String) => Tuple::Value(String(u.to_string())),
          (Tuple::Value(U128(u)), T::String) => Tuple::Value(String(u.to_string())),
          (Tuple::Value(USize(u)), T::String) => Tuple::Value(String(u.to_string())),
          (Tuple::Value(F32(f)), T::String) => Tuple::Value(String(f.to_string())),
          (Tuple::Value(F64(f)), T::String) => Tuple::Value(String(f.to_string())),
          (Tuple::Value(Bool(b)), T::String) => Tuple::Value(String(b.to_string())),
          (Tuple::Value(Char(c)), T::String) => Tuple::Value(String(c.to_string())),
          (Tuple::Value(Str(s)), T::String) => Tuple::Value(String(s.to_string())),
          (Tuple::Value(String(s)), T::String) => Tuple::Value(String(s.clone())),

          // Not implemented
          (v, t) => unimplemented!("Unimplemented type cast from `{:?}` to `{}`", v.tuple_type(), t),
        }
      }
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct IfThenElseExpr {
  pub cond: Box<Expr>,
  pub then_br: Box<Expr>,
  pub else_br: Box<Expr>,
}

impl IfThenElseExpr {
  pub fn new(cond: Expr, then_br: Expr, else_br: Expr) -> Self {
    Self {
      cond: Box::new(cond),
      then_br: Box::new(then_br),
      else_br: Box::new(else_br),
    }
  }

  pub fn eval(&self, v: &Tuple) -> Tuple {
    if self.cond.eval(v).as_bool() {
      self.then_br.eval(v)
    } else {
      self.else_br.eval(v)
    }
  }
}
