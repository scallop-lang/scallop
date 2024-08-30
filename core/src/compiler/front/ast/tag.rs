use std::collections::*;

use crate::common::expr as common_expr;
use crate::common::input_tag::DynamicInputTag;

use super::*;

/// A tag associated with a fact
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub enum Tag {
  Constant(ConstantTag),
  Expr(ExprTag),
}

impl Tag {
  pub fn none() -> Self {
    Self::Constant(ConstantTag::none())
  }

  pub fn is_some(&self) -> bool {
    match self.as_constant() {
      Some(c) => c.is_some(),
      None => false,
    }
  }

  pub fn used_variables(&self) -> BTreeSet<String> {
    match self {
      Self::Constant(_) => BTreeSet::new(),
      Self::Expr(e) => e.used_variables(),
    }
  }

  pub fn to_base_expr(&self, vars: &HashMap<String, usize>) -> Option<common_expr::Expr> {
    match self {
      Self::Constant(c) => c.to_base_expr(vars),
      Self::Expr(e) => e.to_base_expr(vars),
    }
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ConstantTag {
  pub tag: DynamicInputTag,
}

impl ConstantTag {
  pub fn none() -> Self {
    Self::new(DynamicInputTag::None)
  }

  pub fn float(f: f64) -> Self {
    Self::new(DynamicInputTag::Float(f))
  }

  pub fn boolean(b: bool) -> Self {
    Self::new(DynamicInputTag::Bool(b))
  }

  pub fn is_some(&self) -> bool {
    self.tag().is_some()
  }

  pub fn to_base_expr(&self, _vars: &HashMap<String, usize>) -> Option<common_expr::Expr> {
    match self.tag() {
      DynamicInputTag::Bool(b) => Some(common_expr::Expr::constant(*b)),
      DynamicInputTag::Float(f) => Some(common_expr::Expr::constant(*f)),
      _ => None,
    }
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub enum ExprTag {
  Variable(VariableTag),
  Binary(BinaryExprTag),
  Unary(UnaryExprTag),
}

impl ExprTag {
  pub fn used_variables(&self) -> BTreeSet<String> {
    match self {
      Self::Variable(v) => v.used_variables(),
      Self::Binary(b) => b.used_variables(),
      Self::Unary(u) => u.used_variables(),
    }
  }

  pub fn to_base_expr(&self, vars: &HashMap<String, usize>) -> Option<common_expr::Expr> {
    match self {
      Self::Variable(v) => v.to_base_expr(vars),
      Self::Binary(b) => b.to_base_expr(vars),
      Self::Unary(u) => u.to_base_expr(vars),
    }
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _VariableTag {
  pub variable: Identifier,
}

impl VariableTag {
  pub fn used_variables(&self) -> BTreeSet<String> {
    std::iter::once(self.variable().name().clone()).collect()
  }

  pub fn to_base_expr(&self, vars: &HashMap<String, usize>) -> Option<common_expr::Expr> {
    let id = vars.get(self.variable().name())?;
    Some(common_expr::Expr::access(*id))
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _BinaryExprTag {
  pub op: BinaryOp,
  pub op1: Box<Tag>,
  pub op2: Box<Tag>,
}

impl BinaryExprTag {
  pub fn used_variables(&self) -> BTreeSet<String> {
    let left = self.op1().used_variables().into_iter();
    let right = self.op2().used_variables().into_iter();
    left.chain(right).collect()
  }

  pub fn to_base_expr(&self, vars: &HashMap<String, usize>) -> Option<common_expr::Expr> {
    let op1 = self.op1().to_base_expr(vars)?;
    let op2 = self.op2().to_base_expr(vars)?;
    Some(common_expr::Expr::binary(self.op().op().clone(), op1, op2))
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _UnaryExprTag {
  pub op: UnaryOp,
  pub op1: Box<Tag>,
}

impl UnaryExprTag {
  pub fn used_variables(&self) -> BTreeSet<String> {
    self.op1().used_variables()
  }

  pub fn to_base_expr(&self, vars: &HashMap<String, usize>) -> Option<common_expr::Expr> {
    let op = self.op().to_common_unary_op()?;
    let op1 = self.op1().to_base_expr(vars)?;
    Some(common_expr::Expr::unary(op, op1))
  }
}
