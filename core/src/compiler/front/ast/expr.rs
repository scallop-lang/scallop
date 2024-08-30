use crate::common;

use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
pub enum Expr {
  Constant(Constant),
  Variable(Variable),
  Wildcard(Wildcard),
  Binary(BinaryExpr),
  Unary(UnaryExpr),
  IfThenElse(IfThenElseExpr),
  Call(CallExpr),
  New(NewExpr),
  Destruct(DestructExpr),
}

impl Expr {
  pub fn is_complex_expr(&self) -> bool {
    match self {
      Self::Binary(_) | Self::Unary(_) => true,
      _ => false,
    }
  }

  /// Checks if the expression has variables (or wildcards) inside of it
  pub fn has_variable(&self) -> bool {
    match self {
      Self::Constant(_) => false,
      Self::Variable(_) => true,
      Self::Wildcard(_) => true,
      Self::Binary(b) => b.op1().has_variable() || b.op2().has_variable(),
      Self::Unary(b) => b.op1().has_variable(),
      Self::IfThenElse(i) => i.cond().has_variable() || i.then_br().has_variable() || i.else_br().has_variable(),
      Self::Call(c) => c.iter_args().any(|a| a.has_variable()),
      Self::New(n) => n.iter_args().any(|a| a.has_variable()),
      Self::Destruct(n) => n.iter_args().any(|a| a.has_variable()),
    }
  }

  pub fn collect_used_variables(&self) -> Vec<Variable> {
    let mut vars = vec![];
    self.collect_used_variables_helper(&mut vars);
    vars
  }

  fn collect_used_variables_helper(&self, vars: &mut Vec<Variable>) {
    match self {
      Self::Binary(b) => {
        b.op1().collect_used_variables_helper(vars);
        b.op2().collect_used_variables_helper(vars);
      }
      Self::Unary(u) => {
        u.op1().collect_used_variables_helper(vars);
      }
      Self::Call(c) => {
        for a in c.iter_args() {
          a.collect_used_variables_helper(vars);
        }
      }
      Self::Constant(_) => {}
      Self::Wildcard(_) => {}
      Self::IfThenElse(i) => {
        i.cond().collect_used_variables_helper(vars);
        i.then_br().collect_used_variables_helper(vars);
        i.else_br().collect_used_variables_helper(vars);
      }
      Self::Variable(v) => {
        vars.push(v.clone());
      }
      Self::New(n) => {
        for a in n.iter_args() {
          a.collect_used_variables_helper(vars);
        }
      }
      Self::Destruct(n) => {
        for a in n.iter_args() {
          a.collect_used_variables_helper(vars);
        }
      }
    }
  }

  pub fn get_first_variable_location(&self) -> Option<&NodeLocation> {
    match self {
      Expr::Constant(_) => None,
      Expr::Variable(v) => Some(v.location()),
      Expr::Wildcard(w) => Some(w.location()),
      Expr::Binary(b) => Some(b.location()),
      Expr::Unary(u) => u.op1().get_first_variable_location(),
      Expr::IfThenElse(i) => i
        .cond()
        .get_first_variable_location()
        .or_else(|| i.then_br().get_first_variable_location())
        .or_else(|| i.else_br().get_first_variable_location()),
      Expr::Call(c) => {
        for arg in c.iter_args() {
          if let Some(loc) = arg.get_first_variable_location() {
            return Some(loc);
          }
        }
        None
      }
      Expr::New(n) => {
        for arg in n.iter_args() {
          if let Some(loc) = arg.get_first_variable_location() {
            return Some(loc);
          }
        }
        None
      }
      Expr::Destruct(n) => {
        for arg in n.iter_args() {
          if let Some(loc) = arg.get_first_variable_location() {
            return Some(loc);
          }
        }
        None
      }
    }
  }

  pub fn get_first_non_constant_location<F>(&self, is_constant: &F) -> Option<&NodeLocation>
  where
    F: Fn(&Variable) -> bool,
  {
    match self {
      Expr::Constant(_) => None,
      Expr::Variable(v) => {
        if is_constant(v) {
          None
        } else {
          Some(self.location())
        }
      }
      _ => Some(self.location()),
    }
  }
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _Variable {
  pub name: Identifier,
}

impl Variable {
  pub fn variable_name(&self) -> &String {
    self.name().name()
  }
}

impl std::fmt::Display for Variable {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(self.variable_name())
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _VariableBinding {
  pub name: Identifier,
  pub ty: Option<Type>,
}

impl VariableBinding {
  pub fn variable_name(&self) -> &String {
    self.name().name()
  }

  pub fn to_variable(&self) -> Variable {
    Variable::new_with_loc(self.name().clone(), self.location().clone())
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _Wildcard;

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _BinaryOp {
  pub op: crate::common::binary_op::BinaryOp,
}

impl _BinaryOp {
  pub fn add() -> Self {
    Self::new(crate::common::binary_op::BinaryOp::Add)
  }

  pub fn sub() -> Self {
    Self::new(crate::common::binary_op::BinaryOp::Sub)
  }

  pub fn mul() -> Self {
    Self::new(crate::common::binary_op::BinaryOp::Mul)
  }

  pub fn div() -> Self {
    Self::new(crate::common::binary_op::BinaryOp::Div)
  }

  pub fn modulo() -> Self {
    Self::new(crate::common::binary_op::BinaryOp::Mod)
  }

  pub fn and() -> Self {
    Self::new(crate::common::binary_op::BinaryOp::And)
  }

  pub fn or() -> Self {
    Self::new(crate::common::binary_op::BinaryOp::Or)
  }

  pub fn xor() -> Self {
    Self::new(crate::common::binary_op::BinaryOp::Xor)
  }

  pub fn eq() -> Self {
    Self::new(crate::common::binary_op::BinaryOp::Eq)
  }

  pub fn neq() -> Self {
    Self::new(crate::common::binary_op::BinaryOp::Neq)
  }

  pub fn lt() -> Self {
    Self::new(crate::common::binary_op::BinaryOp::Lt)
  }

  pub fn leq() -> Self {
    Self::new(crate::common::binary_op::BinaryOp::Leq)
  }

  pub fn gt() -> Self {
    Self::new(crate::common::binary_op::BinaryOp::Gt)
  }

  pub fn geq() -> Self {
    Self::new(crate::common::binary_op::BinaryOp::Geq)
  }
}

impl BinaryOp {
  pub fn new_add() -> Self {
    Self::new(crate::common::binary_op::BinaryOp::Add)
  }

  pub fn new_eq() -> Self {
    Self::new(crate::common::binary_op::BinaryOp::Eq)
  }

  pub fn is_arith(&self) -> bool {
    self.op().is_arith()
  }

  pub fn is_add_sub(&self) -> bool {
    self.op().is_add_sub()
  }

  pub fn is_logical(&self) -> bool {
    self.op().is_logical()
  }

  pub fn is_eq_neq(&self) -> bool {
    self.op().is_eq_neq()
  }

  pub fn is_eq(&self) -> bool {
    self.op().is_eq()
  }

  pub fn is_numeric_cmp(&self) -> bool {
    self.op().is_numeric_cmp()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _BinaryExpr {
  pub op: BinaryOp,
  pub op1: Box<Expr>,
  pub op2: Box<Expr>,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub enum _UnaryOp {
  Neg,
  Pos,
  Not,
  TypeCast(Type),
}

impl std::fmt::Display for _UnaryOp {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Neg => f.write_str("-"),
      Self::Pos => f.write_str("+"),
      Self::Not => f.write_str("!"),
      Self::TypeCast(ty) => f.write_fmt(format_args!("as {}", ty)),
    }
  }
}

impl UnaryOp {
  pub fn is_pos_neg(&self) -> bool {
    self.is_pos() || self.is_neg()
  }

  /// Cast the AST unary operator to Common unary operator.
  /// The `TypeCast` operation will be discarded.
  pub fn to_common_unary_op(&self) -> Option<common::unary_op::UnaryOp> {
    match self._node {
      _UnaryOp::Neg => Some(common::unary_op::UnaryOp::Neg),
      _UnaryOp::Pos => Some(common::unary_op::UnaryOp::Pos),
      _UnaryOp::Not => Some(common::unary_op::UnaryOp::Not),
      _UnaryOp::TypeCast(_) => None,
    }
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _UnaryExpr {
  pub op: UnaryOp,
  pub op1: Box<Expr>,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _IfThenElseExpr {
  pub cond: Box<Expr>,
  pub then_br: Box<Expr>,
  pub else_br: Box<Expr>,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _CallExpr {
  pub function_identifier: FunctionIdentifier,
  pub args: Vec<Expr>,
}

/// The identifier of a function, i.e. `$abs`
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _FunctionIdentifier {
  pub id: Identifier,
}

impl FunctionIdentifier {
  pub fn name(&self) -> &str {
    self.id().name()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _NewExpr {
  pub functor: Identifier,
  pub args: Vec<Expr>,
}

impl NewExpr {
  pub fn functor_name(&self) -> &str {
    self.functor().name()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _DestructExpr {
  pub functor: Identifier,
  pub args: Vec<Expr>,
}

impl DestructExpr {
  pub fn functor_name(&self) -> &str {
    self.functor().name()
  }
}
