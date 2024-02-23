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
      Self::Atom(a) => Self::NegAtom(NegAtom::new(a.clone())),
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
