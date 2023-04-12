use super::*;

/// A formula
#[derive(Clone, Debug, PartialEq)]
pub enum Formula {
  Atom(Atom),
  NegAtom(NegAtom),
  Disjunction(Disjunction),
  Conjunction(Conjunction),
  Implies(Implies),
  Constraint(Constraint),
  Reduce(Reduce),
  ForallExistsReduce(ForallExistsReduce),
}

impl Formula {
  pub fn conjunction(args: Vec<Self>) -> Self {
    Self::Conjunction(ConjunctionNode { args }.into())
  }

  pub fn negate(&self) -> Self {
    match self {
      Self::Atom(a) => {
        Self::NegAtom(NegAtom::new(a.location().clone(), NegAtomNode { atom: a.clone() }))
      }
      Self::NegAtom(n) => {
        Self::Atom(n.atom().clone())
      },
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
      Self::ForallExistsReduce(_) => {
        // TODO
        panic!("Cannot have aggregation inside negation")
      }
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct AtomNode {
  pub predicate: Identifier,
  pub args: Vec<Expr>,
}

/// A single atom, e.g. `color(x, "blue")`
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
#[doc(hidden)]
pub struct NegAtomNode {
  pub atom: Atom,
}

/// A negated atom, e.g. `not color(x, "blue")`
pub type NegAtom = AstNode<NegAtomNode>;

impl NegAtom {
  pub fn atom(&self) -> &Atom {
    &self.node.atom
  }

  pub fn predicate(&self) -> &String {
    self.atom().predicate()
  }
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct ConjunctionNode {
  pub args: Vec<Formula>,
}

/// A conjunction (AND) formula, e.g. `color(x, "blue") and shape(x, "sphere")`
pub type Conjunction = AstNode<ConjunctionNode>;

impl Conjunction {
  pub fn args(&self) -> impl Iterator<Item = &Formula> {
    self.node.args.iter()
  }
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct DisjunctionNode {
  pub args: Vec<Formula>,
}

/// A disjunction (OR) formula, e.g. `male(x) or female(x)`
pub type Disjunction = AstNode<DisjunctionNode>;

impl Disjunction {
  pub fn args(&self) -> impl Iterator<Item = &Formula> {
    self.node.args.iter()
  }
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct ImpliesNode {
  pub left: Box<Formula>,
  pub right: Box<Formula>,
}

/// An implies formula, e.g. `person(p) implies father(p, _)`
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
#[doc(hidden)]
pub struct ConstraintNode {
  pub expr: Expr,
}

impl ConstraintNode {
  pub fn new(expr: Expr) -> Self {
    Self { expr }
  }
}

/// A constraint, e.g. `x == y`
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

/// A variable or a wildcard, e.g. `x` or `_`
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
#[doc(hidden)]
pub struct ReduceNode {
  pub left: Vec<VariableOrWildcard>,
  pub operator: ReduceOperator,
  pub args: Vec<Variable>,
  pub bindings: Vec<VariableBinding>,
  pub body: Box<Formula>,
  pub group_by: Option<(Vec<VariableBinding>, Box<Formula>)>,
}

/// An aggregation operation, e.g. `n = count(p: person(p))`
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
#[doc(hidden)]
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
  CategoricalK(usize),
  Unknown(String),
}

impl ReduceOperatorNode {
  pub fn to_string(&self) -> String {
    match self {
      Self::Count => "count".to_string(),
      Self::Sum => "sum".to_string(),
      Self::Prod => "prod".to_string(),
      Self::Min => "min".to_string(),
      Self::Max => "max".to_string(),
      Self::Exists => "exists".to_string(),
      Self::Forall => "forall".to_string(),
      Self::Unique => "unique".to_string(),
      Self::TopK(k) => format!("top<{}>", k),
      Self::CategoricalK(k) => format!("categorical<{}>", k),
      Self::Unknown(_) => "unknown".to_string(),
    }
  }
}

/// A reduce opeartor, e.g. `count`
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
      ReduceOperatorNode::CategoricalK(_) => None,
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
    self.node.to_string()
  }
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct ForallExistsReduceNode {
  pub operator: ReduceOperator,
  pub bindings: Vec<VariableBinding>,
  pub body: Box<Formula>,
  pub group_by: Option<(Vec<VariableBinding>, Box<Formula>)>,
}

/// An syntax sugar for forall/exists operation, e.g. `forall(p: person(p) => father(p, _))`.
/// In this case, the assigned variable is omitted for abbrevity.
pub type ForallExistsReduce = AstNode<ForallExistsReduceNode>;

impl ForallExistsReduce {
  pub fn operator(&self) -> &ReduceOperator {
    &self.node.operator
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
