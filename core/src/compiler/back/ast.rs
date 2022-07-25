use std::collections::*;

use super::Attributes;
use crate::common::aggregate_op::AggregateOp;
use crate::common::input_tag::InputTag;
use crate::common::output_option::OutputOption;
use crate::compiler::front;

pub type Type = crate::common::value_type::ValueType;

pub type Constant = crate::common::value::Value;

#[derive(Clone, Debug, PartialEq)]
pub struct Program {
  pub relations: Vec<Relation>,
  pub outputs: HashMap<String, OutputOption>,
  pub facts: Vec<Fact>,
  pub disjunctive_facts: Vec<Vec<Fact>>,
  pub rules: Vec<Rule>,
}

impl Program {
  pub fn relation_of_predicate(&self, pred: &String) -> Option<&Relation> {
    self.relations.iter().find(|r| &r.predicate == pred)
  }

  pub fn rules_of_predicate(&self, pred: String) -> impl Iterator<Item = &Rule> {
    self
      .rules
      .iter()
      .filter(move |r| r.head_predicate() == &pred)
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Fact {
  pub tag: InputTag,
  pub predicate: String,
  pub args: Vec<Constant>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Relation {
  pub attributes: Attributes,
  pub predicate: String,
  pub arg_types: Vec<Type>,
}

impl Relation {
  pub fn new(predicate: String, arg_types: Vec<Type>) -> Self {
    Self::new_with_attrs(Attributes::new(), predicate, arg_types)
  }

  pub fn new_with_attrs(attributes: Attributes, predicate: String, arg_types: Vec<Type>) -> Self {
    Self {
      attributes,
      predicate,
      arg_types,
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
  pub attributes: Attributes,
  pub head: Head,
  pub body: Conjunction,
}

impl Rule {
  pub fn head_predicate(&self) -> &String {
    &self.head.predicate
  }

  pub fn body_literals(&self) -> impl Iterator<Item = &Literal> {
    self.body.args.iter()
  }

  pub fn body_literals_mut(&mut self) -> impl Iterator<Item = &mut Literal> {
    self.body.args.iter_mut()
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Head {
  pub predicate: String,
  pub args: Vec<Term>,
}

impl Head {
  pub fn new(predicate: String, args: Vec<Term>) -> Self {
    Self { predicate, args }
  }

  pub fn variable_args(&self) -> impl Iterator<Item = &Variable> {
    self.args.iter().filter_map(|a| match a {
      Term::Variable(v) => Some(v),
      _ => None,
    })
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Conjunction {
  pub args: Vec<Literal>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Term {
  Variable(Variable),
  Constant(Constant),
}

impl Term {
  pub fn variable(name: String, ty: Type) -> Self {
    Self::Variable(Variable { name, ty })
  }

  pub fn is_variable(&self) -> bool {
    match self {
      Self::Variable(_) => true,
      _ => false,
    }
  }

  pub fn is_constant(&self) -> bool {
    match self {
      Self::Constant(_) => true,
      _ => false,
    }
  }

  pub fn as_variable(&self) -> Option<&Variable> {
    match self {
      Self::Variable(v) => Some(v),
      _ => None,
    }
  }
}

#[derive(Clone, Debug, Hash, PartialEq, PartialOrd, Eq, Ord)]
pub struct Variable {
  pub name: String,
  pub ty: Type,
}

impl Variable {
  pub fn new(name: String, ty: Type) -> Self {
    Self { name, ty }
  }
}

#[derive(Clone, PartialEq)]
pub enum Literal {
  Atom(Atom),
  NegAtom(NegAtom),
  Assign(Assign),
  Constraint(Constraint),
  Reduce(Reduce),
  True,
  False,
}

impl std::fmt::Debug for Literal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Atom(a) => std::fmt::Debug::fmt(a, f),
      Self::NegAtom(a) => std::fmt::Debug::fmt(a, f),
      Self::Assign(a) => std::fmt::Debug::fmt(a, f),
      Self::Constraint(a) => std::fmt::Debug::fmt(a, f),
      Self::Reduce(a) => std::fmt::Debug::fmt(a, f),
      Self::True => f.write_str("true"),
      Self::False => f.write_str("false"),
    }
  }
}

impl Literal {
  pub fn binary_expr(left: Variable, op: BinaryExprOp, op1: Term, op2: Term) -> Self {
    Self::Assign(Assign {
      left,
      right: AssignExpr::Binary(BinaryAssignExpr { op, op1, op2 }),
    })
  }

  pub fn unary_expr(left: Variable, op: UnaryExprOp, op1: Term) -> Self {
    Self::Assign(Assign {
      left,
      right: AssignExpr::Unary(UnaryAssignExpr { op, op1 }),
    })
  }

  pub fn if_then_else_expr(left: Variable, cond: Term, then_br: Term, else_br: Term) -> Self {
    Self::Assign(Assign {
      left,
      right: AssignExpr::IfThenElse(IfThenElseAssignExpr {
        cond,
        then_br,
        else_br,
      }),
    })
  }

  pub fn binary_constraint(op: BinaryConstraintOp, op1: Term, op2: Term) -> Self {
    Self::Constraint(Constraint::Binary(BinaryConstraint { op, op1, op2 }))
  }

  pub fn unary_constraint(op: UnaryConstraintOp, op1: Term) -> Self {
    Self::Constraint(Constraint::Unary(UnaryConstraint { op, op1 }))
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Atom {
  pub predicate: String,
  pub args: Vec<Term>,
}

impl Atom {
  pub fn new(predicate: String, args: Vec<Term>) -> Self {
    Self { predicate, args }
  }

  /// An atom is pure means it contains only variable arguments and
  /// all variables are distinct.
  ///
  /// This implies this atom can be used directly as a dataflow.
  /// Otherwise, additional filter and find may apply.
  pub fn is_pure(&self) -> bool {
    let mut existed_args = HashSet::new();
    for a in &self.args {
      match a {
        Term::Variable(v) => {
          if existed_args.contains(v) {
            return false;
          } else {
            existed_args.insert(v.clone());
          }
        }
        _ => {
          return false;
        }
      }
    }
    return true;
  }

  pub fn variable_args(&self) -> impl Iterator<Item = &Variable> {
    self.args.iter().filter_map(|a| match a {
      Term::Variable(v) => Some(v),
      _ => None,
    })
  }

  pub fn unique_variable_args(&self) -> impl Iterator<Item = Variable> {
    self
      .variable_args()
      .cloned()
      .collect::<BTreeSet<_>>()
      .into_iter()
  }

  pub fn has_constant_arg(&self) -> bool {
    self.args.iter().any(|a| a.is_constant())
  }

  pub fn const_var_partition(&self) -> (Vec<(usize, &Constant)>, Vec<(usize, &Variable)>) {
    let (constants, variables): (Vec<_>, Vec<_>) = self
      .args
      .iter()
      .enumerate()
      .partition(|(_, t)| t.is_constant());
    let constants = constants
      .into_iter()
      .map(|(i, c)| match c {
        Term::Constant(c) => (i, c),
        _ => panic!("Should not happen"),
      })
      .collect();
    let variables = variables
      .into_iter()
      .map(|(i, v)| match v {
        Term::Variable(v) => (i, v),
        _ => panic!("Should not happen"),
      })
      .collect();
    (constants, variables)
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NegAtom {
  pub atom: Atom,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Assign {
  pub left: Variable,
  pub right: AssignExpr,
}

impl Assign {
  pub fn variable_args(&self) -> Vec<&Variable> {
    let mut args = vec![];
    match &self.right {
      AssignExpr::Binary(b) => {
        args.extend(b.op1.as_variable().iter());
        args.extend(b.op2.as_variable().iter());
      }
      AssignExpr::Unary(u) => {
        args.extend(u.op1.as_variable().iter());
      }
      AssignExpr::IfThenElse(i) => {
        args.extend(i.cond.as_variable().iter());
        args.extend(i.then_br.as_variable().iter());
        args.extend(i.else_br.as_variable().iter());
      }
    }
    args
  }
}

#[derive(Clone, Debug, PartialEq)]
pub enum AssignExpr {
  Binary(BinaryAssignExpr),
  Unary(UnaryAssignExpr),
  IfThenElse(IfThenElseAssignExpr),
}

pub type BinaryExprOp = crate::common::binary_op::BinaryOp;

#[derive(Clone, Debug, PartialEq)]
pub struct BinaryAssignExpr {
  pub op: BinaryExprOp,
  pub op1: Term,
  pub op2: Term,
}

pub type UnaryExprOp = crate::common::unary_op::UnaryOp;

#[derive(Clone, Debug, PartialEq)]
pub struct UnaryAssignExpr {
  pub op: UnaryExprOp,
  pub op1: Term,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IfThenElseAssignExpr {
  pub cond: Term,
  pub then_br: Term,
  pub else_br: Term,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Constraint {
  Binary(BinaryConstraint),
  Unary(UnaryConstraint),
}

impl Constraint {
  pub fn variable_args(&self) -> Vec<&Variable> {
    let mut args = vec![];
    match self {
      Self::Binary(b) => {
        args.extend(b.op1.as_variable().iter());
        args.extend(b.op2.as_variable().iter());
      }
      Self::Unary(u) => {
        args.extend(u.op1.as_variable().iter());
      }
    }
    args
  }

  pub fn unique_variable_args(&self) -> impl Iterator<Item = &Variable> {
    self
      .variable_args()
      .into_iter()
      .collect::<BTreeSet<_>>()
      .into_iter()
  }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryConstraintOp {
  Eq,
  Neq,
  Lt,
  Leq,
  Gt,
  Geq,
}

impl From<&BinaryExprOp> for Option<BinaryConstraintOp> {
  fn from(op: &BinaryExprOp) -> Self {
    match op {
      BinaryExprOp::Eq => Some(BinaryConstraintOp::Eq),
      BinaryExprOp::Neq => Some(BinaryConstraintOp::Neq),
      BinaryExprOp::Gt => Some(BinaryConstraintOp::Gt),
      BinaryExprOp::Geq => Some(BinaryConstraintOp::Geq),
      BinaryExprOp::Lt => Some(BinaryConstraintOp::Lt),
      BinaryExprOp::Leq => Some(BinaryConstraintOp::Leq),
      _ => None,
    }
  }
}

impl From<&BinaryConstraintOp> for BinaryExprOp {
  fn from(op: &BinaryConstraintOp) -> Self {
    match op {
      BinaryConstraintOp::Eq => Self::Eq,
      BinaryConstraintOp::Neq => Self::Neq,
      BinaryConstraintOp::Gt => Self::Gt,
      BinaryConstraintOp::Geq => Self::Geq,
      BinaryConstraintOp::Lt => Self::Lt,
      BinaryConstraintOp::Leq => Self::Leq,
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BinaryConstraint {
  pub op: BinaryConstraintOp,
  pub op1: Term,
  pub op2: Term,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnaryConstraintOp {
  Not,
}

impl From<&front::UnaryOpNode> for Option<UnaryConstraintOp> {
  fn from(op: &front::UnaryOpNode) -> Self {
    match op {
      front::UnaryOpNode::Not => Some(UnaryConstraintOp::Not),
      _ => None,
    }
  }
}

impl From<&UnaryConstraintOp> for UnaryExprOp {
  fn from(op: &UnaryConstraintOp) -> Self {
    match op {
      UnaryConstraintOp::Not => Self::Not,
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnaryConstraint {
  pub op: UnaryConstraintOp,
  pub op1: Term,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reduce {
  pub op: AggregateOp,
  pub left_vars: Vec<Variable>,
  pub group_by_vars: Vec<Variable>,
  pub other_group_by_vars: Vec<Variable>,
  pub arg_vars: Vec<Variable>,
  pub to_aggregate_vars: Vec<Variable>,
  pub body_formula: Atom,
  pub group_by_formula: Option<Atom>,
}

impl Reduce {
  pub fn new(
    op: AggregateOp,
    left_vars: Vec<Variable>,
    group_by_vars: Vec<Variable>,
    other_group_by_vars: Vec<Variable>,
    arg_vars: Vec<Variable>,
    to_aggregate_vars: Vec<Variable>,
    body_formula: Atom,
    group_by_formula: Option<Atom>,
  ) -> Self {
    Self {
      op,
      left_vars,
      group_by_vars,
      other_group_by_vars,
      arg_vars,
      to_aggregate_vars,
      body_formula,
      group_by_formula,
    }
  }

  pub fn variable_args(&self) -> impl Iterator<Item = &Variable> {
    self
      .left_vars
      .iter()
      .chain(self.arg_vars.iter())
      .chain(self.group_by_vars.iter())
  }
}
