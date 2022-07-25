use std::collections::*;

use super::*;
use crate::common::expr::{BinaryExpr, Expr, IfThenElseExpr, UnaryExpr};
use crate::common::generic_tuple::GenericTuple;
use crate::common::tuple_access::TupleAccessor;
use crate::common::tuple_type::TupleType;

pub type VariableTuple = GenericTuple<Variable>;

impl std::fmt::Debug for VariableTuple {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Value(v) => f.write_fmt(format_args!("{:?}", v)),
      Self::Tuple(t) => f.write_fmt(format_args!(
        "({})",
        t.iter()
          .map(|e| { format!("{:?}", e) })
          .collect::<Vec<_>>()
          .join(", ")
      )),
    }
  }
}

impl VariableTuple {
  pub fn from_vars<T>(vars: T, can_be_singleton: bool) -> Self
  where
    T: Iterator<Item = Variable>,
  {
    let values = vars.map(Self::Value).collect::<Box<[_]>>();
    if values.is_empty() {
      Self::Tuple(Box::new([]))
    } else if values.len() == 1 && can_be_singleton {
      values[0].clone()
    } else {
      Self::Tuple(values)
    }
  }

  pub fn expand<T>(&self, vars: T) -> Self
  where
    T: Iterator<Item = Variable>,
  {
    let filtered_vars = vars.filter_map(|var| {
      if self.accessor_of(&var).is_some() {
        None
      } else {
        Some(Self::Value(var))
      }
    });
    match self {
      Self::Value(v) => Self::Tuple(
        std::iter::once(Self::Value(v.clone()))
          .chain(filtered_vars)
          .collect(),
      ),
      Self::Tuple(t) => Self::Tuple(t.iter().cloned().chain(filtered_vars).collect()),
    }
  }

  pub fn subtuple<T: Iterator<Item = Variable>>(&self, var_set: T) -> Self {
    let self_vars = self.variables();
    Self::Tuple(
      var_set
        .filter(|v| self_vars.contains(v))
        .map(Self::Value)
        .collect(),
    )
  }

  pub fn dedup(&self) -> Self {
    Self::from_vars(self.variables().into_iter(), true)
  }

  pub fn variables_set(&self) -> HashSet<Variable> {
    match self {
      Self::Value(v) => std::iter::once(v.clone()).collect(),
      Self::Tuple(t) => t.iter().flat_map(|e| e.variables().into_iter()).collect(),
    }
  }

  pub fn variables(&self) -> Vec<Variable> {
    match self {
      Self::Value(v) => std::iter::once(v.clone()).collect(),
      Self::Tuple(t) => {
        let mut visited_variables = HashSet::new();
        t.iter()
          .flat_map(|e| {
            let vars = e
              .variables()
              .into_iter()
              .filter(|v| !visited_variables.contains(v))
              .collect::<Vec<_>>();
            visited_variables.extend(vars.clone().into_iter());
            vars.into_iter()
          })
          .collect()
      }
    }
  }

  pub fn matches(&self, atom: &Atom) -> bool {
    self.matches_args(&atom.args)
  }

  pub fn matches_args(&self, args: &Vec<Term>) -> bool {
    match self {
      Self::Tuple(ts) => {
        if ts.len() == args.len() {
          args.iter().zip(ts.iter()).all(|(t1, t2)| match (t1, t2) {
            (Term::Variable(v1), Self::Value(v2)) => v1 == v2,
            _ => false,
          })
        } else {
          false
        }
      }
      _ => false,
    }
  }

  pub fn accessor_of(&self, var: &Variable) -> Option<TupleAccessor> {
    match self {
      Self::Value(v) => {
        if var == v {
          Some(TupleAccessor::empty())
        } else {
          None
        }
      }
      Self::Tuple(ts) => {
        for (i, e) in ts.iter().enumerate() {
          if let Some(acc) = e.accessor_of(var) {
            return Some(acc.prepend(i as i8));
          }
        }
        None
      }
    }
  }

  pub fn projection(&self, args: &Self) -> Expr {
    match args {
      Self::Tuple(ts) => Expr::Tuple(ts.iter().map(|e| self.projection(e)).collect::<Vec<_>>()),
      Self::Value(v) => Expr::Access(self.accessor_of(v).unwrap()),
    }
  }

  pub fn term_to_ram_expr(&self, t: &Term) -> Option<Expr> {
    match t {
      Term::Constant(c) => Some(Expr::Constant(c.clone())),
      Term::Variable(v) => self.accessor_of(v).map(Expr::Access),
    }
  }

  pub fn projection_assigns(&self, args: &Self, assigns: &HashMap<Variable, AssignExpr>) -> Expr {
    match args {
      Self::Tuple(ts) => Expr::Tuple(
        ts.iter()
          .map(|e| self.projection_assigns(e, assigns))
          .collect(),
      ),
      Self::Value(v) => {
        if let Some(acc) = self.accessor_of(v) {
          Expr::Access(acc)
        } else {
          match &assigns[v] {
            AssignExpr::Binary(b) => Expr::Binary(BinaryExpr {
              op: b.op.clone(),
              op1: Box::new(self.term_to_ram_expr(&b.op1).unwrap()),
              op2: Box::new(self.term_to_ram_expr(&b.op2).unwrap()),
            }),
            AssignExpr::Unary(u) => Expr::Unary(UnaryExpr {
              op: u.op.clone(),
              op1: Box::new(self.term_to_ram_expr(&u.op1).unwrap()),
            }),
            AssignExpr::IfThenElse(i) => Expr::IfThenElse(IfThenElseExpr {
              cond: Box::new(self.term_to_ram_expr(&i.cond).unwrap()),
              then_br: Box::new(self.term_to_ram_expr(&i.then_br).unwrap()),
              else_br: Box::new(self.term_to_ram_expr(&i.else_br).unwrap()),
            }),
          }
        }
      }
    }
  }

  pub fn projection_from_var_access(&self, var_access: &HashMap<&Variable, TupleAccessor>) -> Expr {
    match self {
      Self::Tuple(ts) => Expr::Tuple(
        ts.iter()
          .map(|e| e.projection_from_var_access(var_access))
          .collect(),
      ),
      Self::Value(v) => Expr::Access(var_access[v]),
    }
  }

  pub fn permutation(&self, atom: &Atom) -> Permutation {
    match self {
      Self::Tuple(ts) => Permutation::Tuple(ts.iter().map(|e| e.permutation(atom)).collect()),
      Self::Value(vs) => Permutation::Value(
        atom
          .args
          .iter()
          .position(|a| match a {
            Term::Variable(va) => vs == va,
            _ => false,
          })
          .unwrap(),
      ),
    }
  }

  pub fn tuple_type(&self) -> TupleType {
    match self {
      Self::Tuple(ts) => TupleType::Tuple(ts.iter().map(|e| e.tuple_type()).collect()),
      Self::Value(t) => TupleType::Value(t.ty.clone()),
    }
  }
}
