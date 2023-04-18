use super::*;

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct RuleNode {
  pub head: RuleHead,
  pub body: Formula,
}

impl RuleNode {
  pub fn new(head: RuleHead, body: Formula) -> Self {
    Self { head, body }
  }
}

pub type Rule = AstNode<RuleNode>;

impl Rule {
  pub fn head(&self) -> &RuleHead {
    &self.node.head
  }

  pub fn body(&self) -> &Formula {
    &self.node.body
  }
}

impl Into<Vec<Item>> for Rule {
  fn into(self) -> Vec<Item> {
    vec![Item::RelationDecl(
      RelationDeclNode::Rule(
        RuleDeclNode {
          attrs: Attributes::new(),
          tag: Tag::default_none(),
          rule: self,
        }
        .into(),
      )
      .into(),
    )]
  }
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub enum RuleHeadNode {
  Atom(Atom),
  Disjunction(Vec<Atom>),
}

pub type RuleHead = AstNode<RuleHeadNode>;

impl RuleHead {
  pub fn is_atomic(&self) -> bool {
    match &self.node {
      RuleHeadNode::Atom(_) => true,
      RuleHeadNode::Disjunction(_) => false,
    }
  }

  pub fn is_disjunction(&self) -> bool {
    match &self.node {
      RuleHeadNode::Atom(_) => false,
      RuleHeadNode::Disjunction(_) => true,
    }
  }

  pub fn atom(&self) -> Option<&Atom> {
    match &self.node {
      RuleHeadNode::Atom(atom) => Some(atom),
      RuleHeadNode::Disjunction(_) => None,
    }
  }

  pub fn iter_predicates(&self) -> Vec<&String> {
    match &self.node {
      RuleHeadNode::Atom(atom) => vec![atom.predicate()],
      RuleHeadNode::Disjunction(atoms) => atoms.iter().map(|atom| atom.predicate()).collect(),
    }
  }

  pub fn iter_arguments(&self) -> Vec<&Expr> {
    match &self.node {
      RuleHeadNode::Atom(atom) => atom.iter_arguments().collect(),
      RuleHeadNode::Disjunction(atoms) => atoms
        .iter()
        .flat_map(|atom| atom.iter_arguments())
        .collect(),
    }
  }
}

impl From<Atom> for RuleHead {
  fn from(atom: Atom) -> Self {
    let loc = atom.location().clone_without_id();
    Self::new(loc, RuleHeadNode::Atom(atom))
  }
}
