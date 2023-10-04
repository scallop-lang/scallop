use std::fmt::{Display, Formatter, Result};

use super::*;

impl Display for Item {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    match self {
      Self::ImportDecl(id) => id.fmt(f),
      Self::TypeDecl(td) => td.fmt(f),
      Self::ConstDecl(cd) => cd.fmt(f),
      Self::RelationDecl(rd) => rd.fmt(f),
      Self::QueryDecl(qd) => qd.fmt(f),
    }
  }
}

impl Display for ImportDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attrs() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!("import \"{}\"", self.import_file_path()))
  }
}

impl Display for TypeDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    match self {
      TypeDecl::Subtype(s) => s.fmt(f),
      TypeDecl::Alias(s) => s.fmt(f),
      TypeDecl::Relation(s) => s.fmt(f),
      TypeDecl::Enumerate(e) => e.fmt(f),
      TypeDecl::Algebraic(a) => a.fmt(f),
      TypeDecl::Function(t) => t.fmt(f),
    }
  }
}

impl Display for ConstDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attrs() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!("const "))?;
    for (i, const_assign) in self.iter_assignments().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      const_assign.fmt(f)?;
    }
    Ok(())
  }
}

impl Display for ConstAssignment {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_fmt(format_args!("{}", self.name()))?;
    if let Some(ty) = self.ty() {
      f.write_fmt(format_args!(": {}", ty))?;
    }
    f.write_fmt(format_args!(" = {}", self.value()))
  }
}

impl Display for Entity {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    match self {
      Entity::Expr(e) => e.fmt(f),
      Entity::Object(o) => o.fmt(f),
    }
  }
}

impl Display for Object {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_str(self.functor_name())?;
    f.write_str("(")?;
    for (i, arg) in self.iter_args().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      arg.fmt(f)?;
    }
    f.write_str(")")
  }
}

impl Display for RelationDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    match self {
      RelationDecl::Set(s) => s.fmt(f),
      RelationDecl::Fact(a) => a.fmt(f),
      RelationDecl::Rule(r) => r.fmt(f),
    }
  }
}

impl Display for QueryDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attrs() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!("query {}", self.query()))
  }
}

impl Display for Query {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    match self {
      Query::Atom(a) => a.fmt(f),
      Query::Predicate(p) => p.fmt(f),
    }
  }
}

impl Display for AttributeValue {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    match &self {
      AttributeValue::Constant(c) => c.fmt(f),
      AttributeValue::List(l) => {
        f.write_str("[")?;
        for (i, v) in l.iter().enumerate() {
          if i > 0 {
            f.write_str(", ")?;
          }
          v.fmt(f)?;
        }
        f.write_str("]")
      }
      AttributeValue::Tuple(l) => {
        f.write_str("(")?;
        for (i, v) in l.iter().enumerate() {
          if i > 0 {
            f.write_str(", ")?;
          }
          v.fmt(f)?;
        }
        f.write_str(")")
      }
    }
  }
}

impl Display for Attribute {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_fmt(format_args!("@{}", self.name()))?;
    if self.num_args() > 0 {
      f.write_str("(")?;
      for (i, arg) in self.iter_args().enumerate() {
        if i > 0 {
          f.write_str(", ")?;
        }
        match arg {
          AttributeArg::Pos(p) => {
            f.write_fmt(format_args!("{}", p))?;
          }
          AttributeArg::Kw(kw) => {
            f.write_fmt(format_args!("{} = {}", kw.name(), kw.value()))?;
          }
        }
      }
      f.write_str(")")
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
    Display::fmt(self.internal(), f)
  }
}

impl Display for SubtypeDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attrs() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!("type {} <: {}", self.name(), self.subtype_of()))
  }
}

impl Display for AliasTypeDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attrs() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!("type {} = {}", self.name(), self.alias_of()))
  }
}

impl Display for RelationTypeDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attrs() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_str("type ")?;
    for (i, relation_type) in self.iter_rel_types().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      f.write_fmt(format_args!("{}", relation_type))?;
    }
    Ok(())
  }
}

impl Display for RelationType {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_str(self.predicate_name())?;
    f.write_str("(")?;
    for (i, arg_type) in self.iter_arg_types().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      arg_type.fmt(f)?;
    }
    f.write_str(")")
  }
}

impl Display for EnumTypeDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attrs() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!("type {} = ", self.name()))?;
    for (i, member) in self.iter_members().enumerate() {
      if i > 0 {
        f.write_str(" | ")?;
      }
      member.fmt(f)?;
    }
    Ok(())
  }
}

impl Display for EnumTypeMember {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    self.name().fmt(f)?;
    if let Some(assigned_num) = self.assigned_num() {
      f.write_fmt(format_args!(" = {}", assigned_num))?;
    }
    Ok(())
  }
}

impl Display for AlgebraicDataTypeDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attrs() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    let name = self.name();
    f.write_fmt(format_args!("type {name} = "))?;
    for (i, member) in self.iter_variants().enumerate() {
      if i > 0 {
        f.write_str(" | ")?;
      }
      member.fmt(f)?;
    }
    Ok(())
  }
}

impl Display for AlgebraicDataTypeVariant {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    let name = self.constructor_name();
    let args = self.iter_args().map(|t| format!("{t}")).collect::<Vec<_>>().join(", ");
    f.write_fmt(format_args!("{name}({args})"))
  }
}

impl Display for FunctionTypeDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attrs() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!("type ${}(", self.func_name()))?;
    for (i, arg) in self.iter_args().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      f.write_fmt(format_args!("{}", arg))?;
    }
    f.write_fmt(format_args!(") -> {}", self.ret_ty()))?;
    Ok(())
  }
}

impl Display for Identifier {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_str(self.name())
  }
}

impl Display for ConstantSetDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attrs() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!(
      "rel {} = {{{}}}",
      self.name(),
      self
        .set()
        .iter_tuples()
        .map(|t| format!("{}", t))
        .collect::<Vec<_>>()
        .join(", ")
    ))
  }
}

impl Display for FactDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attrs() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!(
      "rel {}({})",
      self.predicate_name(),
      self
        .iter_args()
        .map(|a| format!("{}", a))
        .collect::<Vec<_>>()
        .join(", ")
    ))
  }
}

impl Display for RuleDecl {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    for attr in self.attrs() {
      f.write_fmt(format_args!("{} ", attr))?;
    }
    f.write_fmt(format_args!("rel {}", self.rule()))
  }
}

impl std::fmt::Display for Atom {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.has_type_args() {
      f.write_fmt(format_args!(
        "{}<{}>({})",
        self.predicate(),
        self
          .iter_type_args()
          .map(|t| format!("{}", t))
          .collect::<Vec<_>>()
          .join(", "),
        self
          .iter_args()
          .map(|a| format!("{}", a))
          .collect::<Vec<_>>()
          .join(", ")
      ))
    } else {
      f.write_fmt(format_args!(
        "{}({})",
        self.predicate(),
        self
          .iter_args()
          .map(|a| format!("{}", a))
          .collect::<Vec<_>>()
          .join(", ")
      ))
    }
  }
}

impl Display for ConstantSetTuple {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    if self.tag().is_some() {
      f.write_fmt(format_args!("{}::", self.tag().tag()))?;
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

impl Display for ConstantOrVariable {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    match self {
      Self::Constant(c) => c.fmt(f),
      Self::Variable(v) => v.fmt(f),
    }
  }
}

impl std::fmt::Display for Constant {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Constant::Integer(i) => i.fmt(f),
      Constant::Entity(e) => e.fmt(f),
      Constant::Float(n) => n.fmt(f),
      Constant::Char(c) => c.fmt(f),
      Constant::Boolean(b) => b.fmt(f),
      Constant::String(s) => s.fmt(f),
      Constant::Symbol(s) => s.fmt(f),
      Constant::DateTime(d) => d.fmt(f),
      Constant::Duration(d) => d.fmt(f),
    }
  }
}

impl std::fmt::Display for IntLiteral {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{}", self.int()))
  }
}

impl std::fmt::Display for FloatLiteral {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{}", self.float()))
  }
}

impl std::fmt::Display for BoolLiteral {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{}", self.value()))
  }
}

impl std::fmt::Display for CharLiteral {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("'{}'", self.character()))
  }
}

impl std::fmt::Display for StringLiteral {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("\"{}\"", self.string()))
  }
}

impl std::fmt::Display for SymbolLiteral {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("s\"{}\"", self.symbol()))
  }
}

impl std::fmt::Display for DateTimeLiteral {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("t\"{}\"", self.datetime()))
  }
}

impl std::fmt::Display for DurationLiteral {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("d\"{}\"", self.duration()))
  }
}

impl std::fmt::Display for EntityLiteral {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("e\"{}\"", self.symbol()))
  }
}

impl Display for Rule {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_fmt(format_args!("{} = {}", self.head(), self.body()))
  }
}

impl Display for RuleHead {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    match self {
      Self::Atom(a) => a.fmt(f),
      Self::Conjunction(c) => {
        for (i, a) in c.iter_atoms().enumerate() {
          if i > 0 {
            f.write_str(", ")?;
          }
          a.fmt(f)?;
        }
        Ok(())
      }
      Self::Disjunction(d) => {
        f.write_str("{")?;
        for (i, a) in d.iter_atoms().enumerate() {
          if i > 0 {
            f.write_str("; ")?;
          }
          a.fmt(f)?;
        }
        f.write_str("}")
      }
    }
  }
}

impl Display for Formula {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    match self {
      Self::Atom(a) => a.fmt(f),
      Self::NegAtom(a) => a.fmt(f),
      Self::Case(c) => c.fmt(f),
      Self::Disjunction(a) => a.fmt(f),
      Self::Conjunction(a) => a.fmt(f),
      Self::Implies(i) => i.fmt(f),
      Self::Constraint(a) => a.fmt(f),
      Self::Reduce(a) => a.fmt(f),
      Self::ForallExistsReduce(a) => a.fmt(f),
      Self::Range(a) => a.fmt(f),
    }
  }
}

impl Display for NegAtom {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_fmt(format_args!("not {}", self.atom()))
  }
}

impl Display for Case {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_fmt(format_args!("case {} is {}", self.variable_name(), self.entity()))
  }
}

impl Display for Disjunction {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_fmt(format_args!(
      "({})",
      self
        .iter_args()
        .map(|a| format!("{}", a))
        .collect::<Vec<_>>()
        .join(" or ")
    ))
  }
}

impl Display for Conjunction {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_fmt(format_args!(
      "({})",
      self
        .iter_args()
        .map(|a| format!("{}", a))
        .collect::<Vec<_>>()
        .join(" and ")
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
    println!("Printing reduce");
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
    f.write_str(" := ")?;
    self.operator().fmt(f)?;
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

impl Display for ForallExistsReduce {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    if self.is_negated() {
      f.write_str("not ")?;
    }
    self.operator().fmt(f)?;
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

impl Display for ReduceOp {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_str(self.name().name())?;
    if self.has_parameters() {
      f.write_str("<")?;
      for (i, param) in self.iter_parameters().enumerate() {
        if i > 0 {
          f.write_str(", ")?;
        }
        param.fmt(f)?;
      }
      f.write_str(">")?;
    }
    if *self.has_exclaimation_mark() {
      f.write_str("!")?;
    }
    Ok(())
  }
}

impl Display for Range {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result {
    f.write_str(&self.to_string())
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
    if let Some(ty) = self.ty() {
      f.write_fmt(format_args!("({}: {})", self.name(), ty))
    } else {
      Display::fmt(self.name(), f)
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
      Self::Call(c) => f.write_fmt(format_args!(
        "${}({})",
        c.function_identifier(),
        c.iter_args().map(|a| format!("{}", a)).collect::<Vec<_>>().join(", ")
      )),
      Self::New(n) => f.write_fmt(format_args!(
        "new {}({})",
        n.functor(),
        n.iter_args().map(|a| format!("{}", a)).collect::<Vec<_>>().join(", ")
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
    f.write_fmt(format_args!("{}", self.internal().op))
  }
}

impl std::fmt::Display for BinaryExpr {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("({} {} {})", self.op1(), self.op(), self.op2()))
  }
}

impl std::fmt::Display for UnaryOp {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self.internal() {
      _UnaryOp::Neg => f.write_str("-"),
      _UnaryOp::Pos => f.write_str("+"),
      _UnaryOp::Not => f.write_str("!"),
      _UnaryOp::TypeCast(t) => f.write_fmt(format_args!("as {}", t)),
    }
  }
}

impl std::fmt::Display for UnaryExpr {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let op = &self.op();
    match op.internal() {
      _UnaryOp::TypeCast(_) => f.write_fmt(format_args!("({} {})", self.op1(), op)),
      _ => f.write_fmt(format_args!("({}{})", op, self.op1())),
    }
  }
}

impl std::fmt::Display for FunctionIdentifier {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(self.name())
  }
}
