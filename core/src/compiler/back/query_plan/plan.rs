use std::collections::*;

use super::*;

#[derive(Clone, Debug)]
pub struct Plan {
  pub bounded_vars: HashSet<Variable>,
  pub ram_node: HighRamNode,
}

impl Plan {
  pub fn unit() -> Self {
    Self {
      bounded_vars: HashSet::new(),
      ram_node: HighRamNode::Unit,
    }
  }

  pub fn pretty_print(&self) {
    self.pretty_print_helper(0);
  }

  fn pretty_print_helper(&self, depth: usize) {
    let prefix = vec!["  "; depth].join("");
    match &self.ram_node {
      HighRamNode::Unit => {
        println!("Unit");
      }
      HighRamNode::Reduce(r) => {
        println!("{}Reduce {{{}}}", prefix, r);
      }
      HighRamNode::Ground(a) => {
        println!("{}{}", prefix, a);
      }
      HighRamNode::Project(x, y) => {
        println!(
          "{}Project {{{}}}",
          prefix,
          y.iter().map(|a| format!("{}", a)).collect::<Vec<_>>().join(", ")
        );
        x.pretty_print_helper(depth + 1);
      }
      HighRamNode::Filter(x, y) => {
        println!(
          "{}Filter {{{}}}",
          prefix,
          y.iter().map(|a| format!("{}", a)).collect::<Vec<_>>().join(", ")
        );
        x.pretty_print_helper(depth + 1);
      }
      HighRamNode::Join(x, y) => {
        println!("{}Join", prefix);
        x.pretty_print_helper(depth + 1);
        y.pretty_print_helper(depth + 1);
      }
      HighRamNode::Antijoin(x, y) => {
        println!("{}Antijoin", prefix);
        x.pretty_print_helper(depth + 1);
        y.pretty_print_helper(depth + 1);
      }
      HighRamNode::ForeignPredicateGround(a) => {
        println!("{}ForeignPredicateGround {{{}}}", prefix, a);
      }
      HighRamNode::ForeignPredicateConstraint(x, a) => {
        println!("{}ForeignPredicateConstraint {{{}}}", prefix, a);
        x.pretty_print_helper(depth + 1);
      }
      HighRamNode::ForeignPredicateJoin(x, a) => {
        println!("{}ForeignPredicateJoin {{{}}}", prefix, a);
        x.pretty_print_helper(depth + 1);
      }
    }
  }
}

#[derive(Clone, Debug)]
pub enum HighRamNode {
  Unit,
  Reduce(Reduce),
  Ground(Atom),
  Project(Box<Plan>, Vec<Assign>),
  Filter(Box<Plan>, Vec<Constraint>),
  Join(Box<Plan>, Box<Plan>),
  Antijoin(Box<Plan>, Box<Plan>),
  ForeignPredicateGround(Atom),
  ForeignPredicateConstraint(Box<Plan>, Atom),
  ForeignPredicateJoin(Box<Plan>, Atom),
}

impl HighRamNode {
  /// Create a new FILTER high level ram node
  pub fn filter(p1: Plan, cs: Vec<Constraint>) -> Self {
    Self::Filter(Box::new(p1), cs)
  }

  /// Create a new Project high level ram node
  pub fn project(p1: Plan, assigns: Vec<Assign>) -> Self {
    Self::Project(Box::new(p1), assigns)
  }

  /// Create a new JOIN high level ram node
  pub fn join(p1: Plan, p2: Plan) -> Self {
    Self::Join(Box::new(p1), Box::new(p2))
  }

  /// Create a new ANTIJOIN high level ram node
  pub fn antijoin(p1: Plan, p2: Plan) -> Self {
    Self::Antijoin(Box::new(p1), Box::new(p2))
  }

  /// Create a new Foreign Predicate Join high level ram node
  pub fn foreign_predicate_join(p1: Plan, a: Atom) -> Self {
    Self::ForeignPredicateJoin(Box::new(p1), a)
  }

  pub fn direct_atom(&self) -> Option<&Atom> {
    match self {
      Self::Ground(a) => Some(a),
      Self::Filter(d, _) => d.ram_node.direct_atom(),
      _ => None,
    }
  }
}
