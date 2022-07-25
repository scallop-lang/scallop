use std::fmt::{Display, Formatter, Result};

use super::*;

impl Display for Item {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    match self {
      Self::ImportDecl(id) => Display::fmt(id, f),
      Self::InputDecl(id) => Display::fmt(id, f),
      Self::TypeDecl(td) => Display::fmt(td, f),
      Self::RelationDecl(rd) => Display::fmt(rd, f),
      Self::QueryDecl(qd) => Display::fmt(qd, f),
    }
  }
}

impl Display for ImportDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attributes() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!("import \"{}\"", self.input_file()))
  }
}

impl Display for InputDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attributes() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!(
      "input {}({})",
      self.predicate(),
      self
        .node
        .types
        .iter()
        .map(|t| format!("{}", t))
        .collect::<Vec<_>>()
        .join(", ")
    ))
  }
}

impl Display for TypeDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    match &self.node {
      TypeDeclNode::Subtype(s) => Display::fmt(s, f),
      TypeDeclNode::Alias(s) => Display::fmt(s, f),
      TypeDeclNode::Relation(s) => Display::fmt(s, f),
    }
  }
}

impl Display for RelationDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    match &self.node {
      RelationDeclNode::Set(s) => Display::fmt(s, f),
      RelationDeclNode::Fact(a) => Display::fmt(a, f),
      RelationDeclNode::Rule(r) => Display::fmt(r, f),
    }
  }
}

impl Display for QueryDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attributes() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!("query {}", self.query()))
  }
}

impl Display for Query {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    match &self.node {
      QueryNode::Atom(a) => Display::fmt(a, f),
      QueryNode::Predicate(p) => Display::fmt(p, f),
    }
  }
}

impl Display for Attribute {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_fmt(format_args!("@{}", self.name()))?;
    if self.num_pos_args() + self.num_kw_args() > 0 {
      f.write_str(
        &self
          .node
          .pos_args
          .iter()
          .map(|a| format!("{}", a))
          .collect::<Vec<_>>()
          .join(", "),
      )?;
      if self.num_pos_args() > 0 && self.num_kw_args() > 0 {
        f.write_str(", ")?;
      }
      f.write_str(
        &self
          .node
          .kw_args
          .iter()
          .map(|(n, a)| format!("{} = {}", n, a))
          .collect::<Vec<_>>()
          .join(", "),
      )
    } else {
      Ok(())
    }
  }
}

impl Display for ArgTypeBinding {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    if let Some(name) = self.name() {
      f.write_fmt(format_args!("{} = {}", name, self.ty()))
    } else {
      Display::fmt(self.ty(), f)
    }
  }
}

impl Display for Type {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    Display::fmt(&self.node, f)
  }
}

impl Display for SubtypeDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attributes() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!(
      "type {} <: {}",
      self.name(),
      self.subtype_of()
    ))
  }
}

impl Display for AliasTypeDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attributes() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!("type {} = {}", self.name(), self.alias_of()))
  }
}

impl Display for RelationTypeDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attributes() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!(
      "type {}({})",
      self.predicate(),
      self
        .arg_types()
        .map(|a| format!("{}", a))
        .collect::<Vec<_>>()
        .join(", ")
    ))
  }
}

impl Display for Identifier {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_str(self.name())
  }
}

impl Display for ConstantSetDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attributes() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!(
      "rel {} = {{{}}}",
      self.predicate(),
      self
        .iter_tuples()
        .map(|t| format!("{}", t))
        .collect::<Vec<_>>()
        .join(", ")
    ))
  }
}

impl Display for FactDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attributes() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!(
      "rel {}({})",
      self.predicate(),
      self
        .iter_arguments()
        .map(|a| format!("{}", a))
        .collect::<Vec<_>>()
        .join(", ")
    ))
  }
}

impl Display for RuleDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attributes() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!("rel {}", self.rule()))
  }
}

impl std::fmt::Display for Atom {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!(
      "{}({})",
      self.predicate(),
      self
        .iter_arguments()
        .map(|a| format!("{}", a))
        .collect::<Vec<_>>()
        .join(", ")
    ))
  }
}

impl Display for ConstantSetTuple {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    if self.tag().is_some() {
      f.write_fmt(format_args!("{}::", self.tag().input_tag()))?;
    }
    f.write_fmt(format_args!(
      "({})",
      self
        .iter_constants()
        .map(|c| format!("{}", c))
        .collect::<Vec<_>>()
        .join(", ")
    ))
  }
}

impl std::fmt::Display for Constant {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match &self.node {
      ConstantNode::Integer(i) => f.write_fmt(format_args!("{}", i)),
      ConstantNode::Float(n) => f.write_fmt(format_args!("{}", n)),
      ConstantNode::Char(c) => f.write_fmt(format_args!("'{}'", c)),
      ConstantNode::Boolean(b) => f.write_fmt(format_args!("{}", b)),
      ConstantNode::String(s) => f.write_fmt(format_args!("\"{}\"", s)),
    }
  }
}

impl Display for Rule {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_fmt(format_args!("{} = {}", self.head(), self.body()))
  }
}

impl Display for Formula {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    match self {
      Self::Atom(a) => a.fmt(f),
      Self::NegAtom(a) => a.fmt(f),
      Self::Disjunction(a) => a.fmt(f),
      Self::Conjunction(a) => a.fmt(f),
      Self::Implies(i) => i.fmt(f),
      Self::Constraint(a) => a.fmt(f),
      Self::Reduce(a) => a.fmt(f),
    }
  }
}

impl Display for NegAtom {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_fmt(format_args!("~{}", self.atom()))
  }
}

impl Display for Disjunction {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_fmt(format_args!(
      "({})",
      self
        .args()
        .map(|a| format!("{}", a))
        .collect::<Vec<_>>()
        .join(" \\/ ")
    ))
  }
}

impl Display for Conjunction {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_fmt(format_args!(
      "({})",
      self
        .args()
        .map(|a| format!("{}", a))
        .collect::<Vec<_>>()
        .join(" /\\ ")
    ))
  }
}

impl Display for Implies {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_fmt(format_args!("({} -> {})", self.left(), self.right()))
  }
}

impl Display for Constraint {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    Display::fmt(self.expr(), f)
  }
}

impl Display for Reduce {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    if self.left().len() > 1 {
      f.write_fmt(format_args!(
        "({})",
        self
          .left()
          .iter()
          .map(|v| format!("{}", v))
          .collect::<Vec<_>>()
          .join(", ")
      ))?;
    } else {
      Display::fmt(self.left().iter().next().unwrap(), f)?;
    }
    f.write_str(" = ")?;
    f.write_str(self.operator().to_str())?;
    if !self.args().is_empty() {
      f.write_fmt(format_args!(
        "[{}]",
        self
          .args()
          .iter()
          .map(|a| format!("{}", a))
          .collect::<Vec<_>>()
          .join(", ")
      ))?;
    }
    f.write_fmt(format_args!(
      "({}: {})",
      self
        .bindings()
        .iter()
        .map(|b| format!("{}", b))
        .collect::<Vec<_>>()
        .join(", "),
      self.body()
    ))
  }
}

impl Display for VariableOrWildcard {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    match self {
      Self::Variable(v) => Display::fmt(v, f),
      Self::Wildcard(w) => Display::fmt(w, f),
    }
  }
}

impl Display for VariableBinding {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    if let Some(ty) = &self.node.ty {
      f.write_fmt(format_args!("({}: {})", self.node.name, ty))
    } else {
      Display::fmt(&self.node.name, f)
    }
  }
}

impl std::fmt::Display for Expr {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Constant(c) => f.write_fmt(format_args!("{}", c)),
      Self::Variable(v) => f.write_fmt(format_args!("{}", v.name())),
      Self::Wildcard(w) => std::fmt::Display::fmt(w, f),
      Self::Binary(b) => std::fmt::Display::fmt(b, f),
      Self::Unary(u) => std::fmt::Display::fmt(u, f),
      Self::IfThenElse(i) => f.write_fmt(format_args!(
        "if {} then {} else {}",
        i.cond(),
        i.then_br(),
        i.else_br()
      )),
    }
  }
}

impl std::fmt::Display for Wildcard {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("_")
  }
}

impl std::fmt::Display for BinaryOp {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{}", self.node))
  }
}

impl std::fmt::Display for BinaryExpr {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!(
      "({} {} {})",
      self.op1(),
      self.op(),
      self.op2()
    ))
  }
}

impl std::fmt::Display for UnaryOpNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Neg => f.write_str("-"),
      Self::Pos => f.write_str("+"),
      Self::Not => f.write_str("!"),
      Self::TypeCast(t) => f.write_fmt(format_args!("as {}", t.node)),
    }
  }
}

impl std::fmt::Display for UnaryExpr {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let op = &self.op().node;
    match op {
      UnaryOpNode::TypeCast(_) => f.write_fmt(format_args!("({} {})", self.op1(), op)),
      _ => f.write_fmt(format_args!("({}{})", op, self.op1())),
    }
  }
}
