use super::binary_op::BinaryOp;
use super::tuple_access::TupleAccessor;
use super::unary_op::UnaryOp;
use super::value::Value;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Expr {
  Tuple(Vec<Expr>),
  Access(TupleAccessor),
  Constant(Value),
  Binary(BinaryExpr),
  Unary(UnaryExpr),
  IfThenElse(IfThenElseExpr),
  Call(CallExpr),
}

impl Expr {
  pub fn binary(op: BinaryOp, op1: Expr, op2: Expr) -> Self {
    Self::Binary(BinaryExpr::new(op, op1, op2))
  }

  pub fn unary(op: UnaryOp, op1: Expr) -> Self {
    Self::Unary(UnaryExpr::new(op, op1))
  }

  pub fn ite(c: Expr, t: Expr, e: Expr) -> Self {
    Self::IfThenElse(IfThenElseExpr::new(c, t, e))
  }

  pub fn access<T: Into<TupleAccessor>>(t: T) -> Self {
    Self::Access(t.into())
  }

  pub fn constant<T: Into<Value>>(t: T) -> Self {
    Self::Constant(t.into())
  }

  pub fn call(function: String, args: Vec<Expr>) -> Self {
    Self::Call(CallExpr { function, args })
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

  pub fn compose(&self, other: &Expr) -> Self {
    match (self, other) {
      (Self::Constant(c), _) => Self::Constant(c.clone()),
      (Self::Access(a1), Self::Access(a2)) => Self::Access(a2.join(a1)),
      (Self::Access(a), Self::Tuple(t)) => {
        if a.len() == 0 {
          Self::Tuple(t.clone())
        } else {
          let sub_expr = &t[a.indices[0] as usize];
          let sub_acc = a.shift();
          sub_expr.compose(&Expr::Access(sub_acc))
        }
      }
      (Self::Access(a), e) => {
        if a.len() == 0 {
          e.clone()
        } else {
          panic!("Type mismatch")
        }
      }
      (Self::Tuple(t), e) => Self::Tuple(t.iter().map(|e0| e0.compose(e)).collect()),
      (Self::Binary(b), e) => Self::binary(b.op.clone(), b.op1.compose(e), b.op2.compose(e)),
      (Self::Unary(u), e) => Self::unary(u.op.clone(), u.op1.compose(e)),
      (Self::IfThenElse(i), e) => Self::ite(i.cond.compose(e), i.then_br.compose(e), i.else_br.compose(e)),
      (Self::Call(c), e) => Self::call(c.function.clone(), c.args.iter().map(|a| a.compose(e)).collect()),
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
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CallExpr {
  pub function: String,
  pub args: Vec<Expr>,
}
