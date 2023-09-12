use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ConstantTuple {
  pub elems: Vec<ConstantOrVariable>,
}

impl ConstantTuple {
  pub fn arity(&self) -> usize {
    self.elems().len()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ConstantSetTuple {
  pub tag: Tag,
  pub tuple: ConstantTuple,
}

impl ConstantSetTuple {
  pub fn arity(&self) -> usize {
    self.tuple().arity()
  }

  pub fn iter_constants(&self) -> impl Iterator<Item = &ConstantOrVariable> {
    self.tuple().elems().iter()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ConstantSet {
  pub tuples: Vec<ConstantSetTuple>,
  pub is_disjunction: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ConstantSetDecl {
  pub attrs: Attributes,
  pub name: Identifier,
  pub set: ConstantSet,
}

impl ConstantSetDecl {
  pub fn predicate_name(&self) -> &String {
    self.name().name()
  }

  pub fn is_disjunction(&self) -> bool {
    self.set().is_disjunction().clone()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _FactDecl {
  pub attrs: Attributes,
  pub tag: Tag,
  pub atom: Atom,
}

impl FactDecl {
  pub fn predicate_name(&self) -> &String {
    self.atom().predicate().name()
  }

  pub fn arity(&self) -> usize {
    self.atom().arity()
  }

  pub fn iter_args(&self) -> impl Iterator<Item = &Expr> {
    self.atom().iter_args()
  }

  pub fn iter_constants(&self) -> impl Iterator<Item = &Constant> {
    self.iter_args().filter_map(|expr| expr.as_constant())
  }

  pub fn has_tag(&self) -> bool {
    self.tag().is_some()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _RuleDecl {
  pub attrs: Attributes,
  pub tag: Tag,
  pub rule: Rule,
}

impl RuleDecl {
  pub fn rule_tag_predicate(&self) -> String {
    if let Some(head_atom) = self.rule().head().as_atom() {
      format!("rt#{}#{}", head_atom.predicate(), self.location_id().expect("location id has not been tagged yet"))
    } else {
      unimplemented!("Rule head is not an atom")
    }
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub enum RelationDecl {
  Set(ConstantSetDecl),
  Fact(FactDecl),
  Rule(RuleDecl),
}

impl RelationDecl {
  pub fn attrs(&self) -> &Attributes {
    match self {
      RelationDecl::Set(s) => s.attrs(),
      RelationDecl::Fact(f) => f.attrs(),
      RelationDecl::Rule(r) => r.attrs(),
    }
  }

  pub fn attrs_mut(&mut self) -> &mut Attributes {
    match self {
      RelationDecl::Set(s) => s.attrs_mut(),
      RelationDecl::Fact(f) => f.attrs_mut(),
      RelationDecl::Rule(r) => r.attrs_mut(),
    }
  }
}

impl From<RelationDecl> for Item {
  fn from(q: RelationDecl) -> Self {
    Self::RelationDecl(q)
  }
}
