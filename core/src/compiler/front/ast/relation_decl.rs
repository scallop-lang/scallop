use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantTupleNode {
  pub elems: Vec<ConstantOrVariable>,
}

pub type ConstantTuple = AstNode<ConstantTupleNode>;

impl ConstantTuple {
  pub fn arity(&self) -> usize {
    self.node.elems.len()
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantSetTupleNode {
  pub tag: Tag,
  pub tuple: ConstantTuple,
}

pub type ConstantSetTuple = AstNode<ConstantSetTupleNode>;

impl ConstantSetTuple {
  pub fn tag(&self) -> &Tag {
    &self.node.tag
  }

  pub fn arity(&self) -> usize {
    self.node.tuple.arity()
  }

  pub fn iter_constants(&self) -> impl Iterator<Item = &ConstantOrVariable> {
    self.node.tuple.node.elems.iter()
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantSetNode {
  pub tuples: Vec<ConstantSetTuple>,
  pub is_disjunction: bool,
}

pub type ConstantSet = AstNode<ConstantSetNode>;

impl ConstantSet {
  pub fn num_tuples(&self) -> usize {
    self.node.tuples.len()
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantSetDeclNode {
  pub attrs: Attributes,
  pub name: Identifier,
  pub set: ConstantSet,
}

pub type ConstantSetDecl = AstNode<ConstantSetDeclNode>;

impl ConstantSetDecl {
  pub fn attributes(&self) -> &Attributes {
    &self.node.attrs
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    &mut self.node.attrs
  }

  pub fn predicate(&self) -> &String {
    &self.node.name.node.name
  }

  pub fn is_disjunction(&self) -> bool {
    self.node.set.node.is_disjunction
  }

  pub fn num_tuples(&self) -> usize {
    self.node.set.num_tuples()
  }

  pub fn iter_tuples(&self) -> impl Iterator<Item = &ConstantSetTuple> {
    self.node.set.node.tuples.iter()
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FactDeclNode {
  pub attrs: Attributes,
  pub tag: Tag,
  pub atom: Atom,
}

pub type FactDecl = AstNode<FactDeclNode>;

impl FactDecl {
  pub fn attributes(&self) -> &Attributes {
    &self.node.attrs
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    &mut self.node.attrs
  }

  pub fn predicate(&self) -> &String {
    self.node.atom.predicate()
  }

  pub fn arity(&self) -> usize {
    self.node.atom.arity()
  }

  pub fn iter_arguments(&self) -> impl Iterator<Item = &Expr> {
    self.node.atom.iter_arguments()
  }

  pub fn iter_constants(&self) -> impl Iterator<Item = &Constant> {
    self.iter_arguments().map(|expr| match expr {
      Expr::Constant(c) => c,
      _ => panic!("[Internal Error] Fact argument not constant"),
    })
  }

  pub fn atom(&self) -> &Atom {
    &self.node.atom
  }

  pub fn has_tag(&self) -> bool {
    self.node.tag.is_some()
  }

  pub fn tag(&self) -> &Tag {
    &self.node.tag
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RuleDeclNode {
  pub attrs: Attributes,
  pub tag: Tag,
  pub rule: Rule,
}

impl RuleDeclNode {
  pub fn new(attrs: Attributes, tag: Tag, rule: Rule) -> Self {
    Self { attrs, tag, rule }
  }
}

pub type RuleDecl = AstNode<RuleDeclNode>;

impl RuleDecl {
  pub fn attributes(&self) -> &Attributes {
    &self.node.attrs
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    &mut self.node.attrs
  }

  pub fn rule(&self) -> &Rule {
    &self.node.rule
  }

  pub fn tag(&self) -> &Tag {
    &self.node.tag
  }

  pub fn rule_tag_predicate(&self) -> String {
    format!("rt#{}#{}", self.rule().head().predicate(), self.id())
  }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RelationDeclNode {
  Set(ConstantSetDecl),
  Fact(FactDecl),
  Rule(RuleDecl),
}

pub type RelationDecl = AstNode<RelationDeclNode>;

impl RelationDecl {
  pub fn attributes(&self) -> &Attributes {
    match &self.node {
      RelationDeclNode::Set(s) => s.attributes(),
      RelationDeclNode::Fact(f) => f.attributes(),
      RelationDeclNode::Rule(r) => r.attributes(),
    }
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    match &mut self.node {
      RelationDeclNode::Set(s) => s.attributes_mut(),
      RelationDeclNode::Fact(f) => f.attributes_mut(),
      RelationDeclNode::Rule(r) => r.attributes_mut(),
    }
  }
}

impl From<RelationDecl> for Item {
  fn from(q: RelationDecl) -> Self {
    Self::RelationDecl(q)
  }
}
