use super::*;

/// A formula
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
pub enum Formula {
  Atom(Atom),
  NegAtom(NegAtom),
  Case(Case),
  Disjunction(Disjunction),
  Conjunction(Conjunction),
  Implies(Implies),
  Constraint(Constraint),
  Reduce(Reduce),
  ForallExistsReduce(ForallExistsReduce),
  Range(Range),
}

impl Formula {
  pub fn negate(&self) -> Self {
    match self {
      Self::Atom(a) => {
        Self::NegAtom(NegAtom::new(a.clone()))
      },
      Self::NegAtom(n) => Self::Atom(n.atom().clone()),
      Self::Case(_) => {
        // TODO
        panic!("Cannot have case inside negation")
      }
      Self::Disjunction(d) => Self::conjunction(Conjunction::new(d.iter_args().map(|f| f.negate()).collect())),
      Self::Conjunction(c) => Self::disjunction(Disjunction::new(c.iter_args().map(|f| f.negate()).collect())),
      Self::Implies(i) => Self::conjunction(Conjunction::new(vec![i.left().clone(), i.right().negate()])),
      Self::Constraint(c) => Self::Constraint(c.negate()),
      Self::Reduce(_) => {
        // TODO
        panic!("Cannot have aggregation inside negation")
      }
      Self::ForallExistsReduce(_) => {
        // TODO
        panic!("Cannot have aggregation inside negation")
      }
      Self::Range(_) => {
        panic!("Cannot have range inside negation")
      }
    }
  }
}

/// A single atom, e.g. `color(x, "blue")`
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _Atom {
  pub predicate: Identifier,
  pub type_args: Vec<Type>,
  pub args: Vec<Expr>,
}

impl Atom {
  pub fn formatted_predicate(&self) -> String {
    if self.has_type_args() {
      let args = self
        .iter_type_args()
        .map(|a| format!("{}", a))
        .collect::<Vec<_>>()
        .join("#");
      format!("{}#{}", self.predicate().name(), args)
    } else {
      self.predicate().name().to_string()
    }
  }

  pub fn arity(&self) -> usize {
    self.num_args()
  }
}

/// A negated atom, e.g. `not color(x, "blue")`
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _NegAtom {
  pub atom: Atom,
}

impl NegAtom {
  pub fn predicate(&self) -> &Identifier {
    self.atom().predicate()
  }

  // pub fn predicate_name(&self) -> &String {
  //   self.atom().predicate_name()
  // }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _Case {
  pub variable: Variable,
  pub entity: Entity,
}

impl Case {
  pub fn variable_name(&self) -> &String {
    self.variable().name().name()
  }
}

/// A conjunction (AND) formula, e.g. `color(x, "blue") and shape(x, "sphere")`
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _Conjunction {
  pub args: Vec<Formula>,
}

/// A disjunction (OR) formula, e.g. `male(x) or female(x)`
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _Disjunction {
  pub args: Vec<Formula>,
}

/// An implies formula, e.g. `person(p) implies father(p, _)`
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _Implies {
  pub left: Box<Formula>,
  pub right: Box<Formula>,
}

/// A constraint, e.g. `x == y`
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _Constraint {
  pub expr: Expr,
}

impl Constraint {
  pub fn negate(&self) -> Self {
    unimplemented!("Negating a constraint has not been implemented")
  }
}

/// A variable or a wildcard, e.g. `x` or `_`
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
pub enum VariableOrWildcard {
  Variable(Variable),
  Wildcard(Wildcard),
}

impl VariableOrWildcard {
  pub fn name(&self) -> Option<&String> {
    match self {
      Self::Variable(v) => Some(v.name().name()),
      _ => None,
    }
  }
}

/// An aggregation operation, e.g. `n = count(p: person(p))`
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _Reduce {
  pub left: Vec<VariableOrWildcard>,
  pub operator: ReduceOp,
  pub args: Vec<Variable>,
  pub bindings: Vec<VariableBinding>,
  pub body: Box<Formula>,
  pub group_by: Option<(Vec<VariableBinding>, Box<Formula>)>,
}

impl Reduce {
  pub fn iter_left_variables(&self) -> impl Iterator<Item = &Variable> {
    self.left().iter().filter_map(|i| match i {
      VariableOrWildcard::Variable(v) => Some(v),
      _ => None,
    })
  }

  pub fn iter_binding_names(&self) -> impl Iterator<Item = &String> {
    self.bindings().iter().map(|b| b.name().name())
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub enum _ReduceOp {
  Count(bool),
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

impl ToString for _ReduceOp {
  fn to_string(&self) -> String {
    match self {
      Self::Count(discrete) => if *discrete {
        "count!".to_string()
      } else {
        "count".to_string()
      },
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

impl ReduceOp {
  pub fn output_arity(&self) -> Option<usize> {
    match self.internal() {
      _ReduceOp::Count(_) => Some(1),
      _ReduceOp::Sum => Some(1),
      _ReduceOp::Prod => Some(1),
      _ReduceOp::Min => Some(1),
      _ReduceOp::Max => Some(1),
      _ReduceOp::Exists => Some(1),
      _ReduceOp::Forall => Some(1),
      _ReduceOp::Unique => None,
      _ReduceOp::TopK(_) => None,
      _ReduceOp::CategoricalK(_) => None,
      _ReduceOp::Unknown(_) => None,
    }
  }

  pub fn num_bindings(&self) -> Option<usize> {
    match self.internal() {
      _ReduceOp::Count(_) => None,
      _ReduceOp::Sum => Some(1),
      _ReduceOp::Prod => Some(1),
      _ReduceOp::Min => Some(1),
      _ReduceOp::Max => Some(1),
      _ReduceOp::Exists => None,
      _ReduceOp::Forall => None,
      _ReduceOp::Unique => None,
      _ReduceOp::TopK(_) => None,
      _ => None,
    }
  }
}

impl ToString for ReduceOp {
  fn to_string(&self) -> String {
    self.internal().to_string()
  }
}

/// An syntax sugar for forall/exists operation, e.g. `forall(p: person(p) => father(p, _))`.
/// In this case, the assigned variable is omitted for abbrevity.
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ForallExistsReduce {
  pub negate: bool,
  pub operator: ReduceOp,
  pub bindings: Vec<VariableBinding>,
  pub body: Box<Formula>,
  pub group_by: Option<(Vec<VariableBinding>, Box<Formula>)>,
}

impl ForallExistsReduce {
  pub fn is_negated(&self) -> bool {
    self.negate().clone()
  }

  pub fn iter_binding_names(&self) -> impl Iterator<Item = &String> {
    self.iter_bindings().map(|b| b.name().name())
  }
}

// Syntax sugar for range operation
// `rel grid(x, y) = x in 0..10 and y in 3..=5`
// is equivalent to
// `rel grid(x, y) = range_i32(0, 10, x) and range_i32(3, 6, y)`
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
pub struct _Range {
  pub left: Variable,
  pub lower: Expr,
  pub upper: Expr,
  pub inclusive: bool,
}
