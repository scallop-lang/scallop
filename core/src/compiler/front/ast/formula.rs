use super::*;

#[derive(Clone, Debug, PartialEq)]
pub enum Formula {
  Atom(Atom),
  NegAtom(NegAtom),
  Disjunction(Disjunction),
  Conjunction(Conjunction),
  Implies(Implies),
  Constraint(Constraint),
  Reduce(Reduce),
}

impl Formula {
  pub fn conjunction(args: Vec<Self>) -> Self {
    Self::Conjunction(ConjunctionNode { args }.into())
  }

  pub fn negate(&self) -> Self {
    match self {
      Self::Atom(a) => Self::NegAtom(NegAtom::new(a.location().clone(), NegAtomNode { atom: a.clone() })),
      Self::NegAtom(n) => Self::Atom(n.atom().clone()),
      Self::Disjunction(d) => Self::Conjunction(Conjunction::new(
        d.location().clone(),
        ConjunctionNode {
          args: d.args().map(|f| f.negate()).collect(),
        },
      )),
      Self::Conjunction(c) => Self::Disjunction(Disjunction::new(
        c.location().clone(),
        DisjunctionNode {
          args: c.args().map(|f| f.negate()).collect(),
        },
      )),
      Self::Implies(i) => Self::Conjunction(Conjunction::new(
        i.location().clone(),
        ConjunctionNode {
          args: vec![i.left().clone(), i.right().negate()],
        },
      )),
      Self::Constraint(c) => Self::Constraint(c.negate()),
      Self::Reduce(_) => {
        // TODO
        panic!("Cannot have aggregation inside negation")
      }
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AtomNode {
  pub predicate: Identifier,
  pub args: Vec<Expr>,
}

pub type Atom = AstNode<AtomNode>;

impl Atom {
  pub fn predicate(&self) -> &String {
    &self.node.predicate.node.name
  }

  pub fn arity(&self) -> usize {
    self.node.args.len()
  }

  pub fn iter_arguments(&self) -> impl Iterator<Item = &Expr> {
    self.node.args.iter()
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NegAtomNode {
  pub atom: Atom,
}

pub type NegAtom = AstNode<NegAtomNode>;

impl NegAtom {
  pub fn atom(&self) -> &Atom {
    &self.node.atom
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConjunctionNode {
  pub args: Vec<Formula>,
}

pub type Conjunction = AstNode<ConjunctionNode>;

impl Conjunction {
  pub fn args(&self) -> impl Iterator<Item = &Formula> {
    self.node.args.iter()
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DisjunctionNode {
  pub args: Vec<Formula>,
}

pub type Disjunction = AstNode<DisjunctionNode>;

impl Disjunction {
  pub fn args(&self) -> impl Iterator<Item = &Formula> {
    self.node.args.iter()
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImpliesNode {
  pub left: Box<Formula>,
  pub right: Box<Formula>,
}

pub type Implies = AstNode<ImpliesNode>;

impl Implies {
  pub fn left(&self) -> &Formula {
    &self.node.left
  }

  pub fn right(&self) -> &Formula {
    &self.node.right
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstraintNode {
  pub expr: Expr,
}

impl ConstraintNode {
  pub fn new(expr: Expr) -> Self {
    Self { expr }
  }
}

pub type Constraint = AstNode<ConstraintNode>;

impl Constraint {
  pub fn default_with_expr(expr: Expr) -> Self {
    Self::default(ConstraintNode { expr })
  }

  pub fn expr(&self) -> &Expr {
    &self.node.expr
  }

  pub fn negate(&self) -> Self {
    unimplemented!("Negating a constraint has not been implemented")
  }
}

#[derive(Clone, Debug, PartialEq)]
pub enum VariableOrWildcard {
  Variable(Variable),
  Wildcard(Wildcard),
}

impl VariableOrWildcard {
  pub fn name(&self) -> Option<&str> {
    match self {
      Self::Variable(v) => Some(v.name()),
      _ => None,
    }
  }

  pub fn location(&self) -> &AstNodeLocation {
    match self {
      Self::Variable(v) => v.location(),
      Self::Wildcard(w) => w.location(),
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReduceNode {
  pub left: Vec<VariableOrWildcard>,
  pub operator: ReduceOperator,
  pub args: Vec<Variable>,
  pub bindings: Vec<VariableBinding>,
  pub body: Box<Formula>,
  pub group_by: Option<(Vec<VariableBinding>, Box<Formula>)>,
}

pub type Reduce = AstNode<ReduceNode>;

impl Reduce {
  pub fn operator(&self) -> &ReduceOperator {
    &self.node.operator
  }

  pub fn left(&self) -> &Vec<VariableOrWildcard> {
    &self.node.left
  }

  pub fn left_variables(&self) -> impl Iterator<Item = &Variable> {
    self.node.left.iter().filter_map(|i| match i {
      VariableOrWildcard::Variable(v) => Some(v),
      _ => None,
    })
  }

  pub fn args(&self) -> &Vec<Variable> {
    &self.node.args
  }

  pub fn bindings(&self) -> &Vec<VariableBinding> {
    &self.node.bindings
  }

  pub fn binding_names(&self) -> impl Iterator<Item = &str> {
    self.node.bindings.iter().map(|b| b.name())
  }

  pub fn body(&self) -> &Formula {
    &self.node.body
  }

  pub fn group_by(&self) -> Option<(&Vec<VariableBinding>, &Formula)> {
    self.node.group_by.as_ref().map(|(b, f)| (b, &**f))
  }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReduceOperatorNode {
  Count,
  Sum,
  Prod,
  Min,
  Max,
  Exists,
  Forall,
  Unique,
  TopK(usize),
  Unknown(String),
}

pub type ReduceOperator = AstNode<ReduceOperatorNode>;

impl ReduceOperator {
  pub fn is_forall(&self) -> bool {
    match &self.node {
      ReduceOperatorNode::Forall => true,
      _ => false,
    }
  }

  pub fn output_arity(&self) -> Option<usize> {
    match &self.node {
      ReduceOperatorNode::Count => Some(1),
      ReduceOperatorNode::Sum => Some(1),
      ReduceOperatorNode::Prod => Some(1),
      ReduceOperatorNode::Min => Some(1),
      ReduceOperatorNode::Max => Some(1),
      ReduceOperatorNode::Exists => Some(1),
      ReduceOperatorNode::Forall => Some(1),
      ReduceOperatorNode::Unique => None,
      ReduceOperatorNode::TopK(_) => None,
      ReduceOperatorNode::Unknown(_) => None,
    }
  }

  pub fn num_bindings(&self) -> Option<usize> {
    match &self.node {
      ReduceOperatorNode::Count => None,
      ReduceOperatorNode::Sum => Some(1),
      ReduceOperatorNode::Prod => Some(1),
      ReduceOperatorNode::Min => Some(1),
      ReduceOperatorNode::Max => Some(1),
      ReduceOperatorNode::Exists => None,
      ReduceOperatorNode::Forall => None,
      ReduceOperatorNode::Unique => None,
      ReduceOperatorNode::TopK(_) => None,
      _ => None,
    }
  }

  pub fn to_string(&self) -> String {
    match &self.node {
      ReduceOperatorNode::Count => "count".to_string(),
      ReduceOperatorNode::Sum => "sum".to_string(),
      ReduceOperatorNode::Prod => "prod".to_string(),
      ReduceOperatorNode::Min => "min".to_string(),
      ReduceOperatorNode::Max => "max".to_string(),
      ReduceOperatorNode::Exists => "exists".to_string(),
      ReduceOperatorNode::Forall => "forall".to_string(),
      ReduceOperatorNode::Unique => "unique".to_string(),
      ReduceOperatorNode::TopK(k) => format!("top<{}>", k),
      ReduceOperatorNode::Unknown(_) => "unknown".to_string(),
    }
  }
}
