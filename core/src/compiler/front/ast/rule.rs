use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _Rule {
  pub head: RuleHead,
  pub body: Formula,
}

impl Into<Vec<Item>> for Rule {
  fn into(self) -> Vec<Item> {
    vec![Item::RelationDecl(RelationDecl::Rule(RuleDecl::new(
      Attributes::new(),
      Tag::none(),
      self,
    )))]
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ConjunctiveRuleHead {
  pub atoms: Vec<Atom>,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _DisjunctiveRuleHead {
  pub atoms: Vec<Atom>,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub enum RuleHead {
  Atom(Atom),
  Conjunction(ConjunctiveRuleHead),
  Disjunction(DisjunctiveRuleHead),
}

impl RuleHead {
  pub fn iter_predicates(&self) -> Vec<String> {
    match self {
      RuleHead::Atom(atom) => vec![atom.predicate().name().clone()],
      RuleHead::Conjunction(conj) => conj.iter_atoms().map(|atom| atom.predicate().name()).cloned().collect(),
      RuleHead::Disjunction(disj) => disj.iter_atoms().map(|atom| atom.predicate().name()).cloned().collect(),
    }
  }

  pub fn iter_args(&self) -> Vec<&Expr> {
    match self {
      RuleHead::Atom(atom) => atom.iter_args().collect(),
      RuleHead::Conjunction(conj) => conj.iter_atoms().flat_map(|atom| atom.iter_args()).collect(),
      RuleHead::Disjunction(disj) => disj.iter_atoms().flat_map(|atom| atom.iter_args()).collect(),
    }
  }
}

impl From<Atom> for RuleHead {
  fn from(atom: Atom) -> Self {
    Self::Atom(atom)
  }
}
