use crate::runtime::env::Scheduler;

use super::*;
use std::fmt::{Display, Formatter, Result as FmtResult};

impl Display for Program {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_str("Relations:")?;
    for relation in &self.relations {
      f.write_fmt(format_args!("\n  {}", relation))?;
    }
    f.write_str("\nFacts:")?;
    for fact in &self.facts {
      f.write_fmt(format_args!("\n  {}", fact))?;
    }
    f.write_str("\nDisjunctive Facts:")?;
    for (i, disjunction) in self.disjunctive_facts.iter().enumerate() {
      f.write_fmt(format_args!("\n  Disjunction {}:", i))?;
      for fact in disjunction {
        f.write_fmt(format_args!("\n    {}", fact))?;
      }
    }
    f.write_str("\nRules:")?;
    for rule in &self.rules {
      f.write_fmt(format_args!("\n  {}", rule))?;
    }
    Ok(())
  }
}

impl Display for Relation {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    self.attributes.fmt(f)?;
    f.write_fmt(format_args!(" {}(", self.predicate))?;
    for (i, arg) in self.arg_types.iter().enumerate() {
      arg.fmt(f)?;
      if i < self.arg_types.len() - 1 {
        f.write_str(", ")?;
      }
    }
    f.write_str(")")
  }
}

impl Display for Attributes {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    for attr in &self.attrs {
      attr.fmt(f)?;
    }
    Ok(())
  }
}

impl Display for Attribute {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    match self {
      Self::AggregateBody(a) => a.fmt(f),
      Self::AggregateGroupBy(a) => a.fmt(f),
      Self::Demand(d) => d.fmt(f),
      Self::MagicSet(d) => d.fmt(f),
      Self::InputFile(i) => i.fmt(f),
      Self::Goal(g) => g.fmt(f),
      Self::Scheduler(s) => s.fmt(f),
    }
  }
}

impl Display for AggregateBodyAttribute {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_fmt(format_args!(
      "@aggregate_body(num_group_by = {}, num_arg = {}, num_key: {}) ",
      self.num_group_by_vars, self.num_arg_vars, self.num_key_vars
    ))
  }
}

impl Display for AggregateGroupByAttribute {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_fmt(format_args!(
      "@aggregate_group_by(num_joined = {}, num_other = {})",
      self.num_join_group_by_vars, self.num_other_group_by_vars,
    ))
  }
}

impl Display for DemandAttribute {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_fmt(format_args!("@demand({:?})", self.pattern))
  }
}

impl Display for MagicSetAttribute {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_str("@magic_set")
  }
}

impl Display for InputFileAttribute {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_fmt(format_args!("@file({:?})", self.input_file))
  }
}

impl Display for GoalAttribute {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_str("@goal")
  }
}

impl Display for SchedulerAttribute {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_str("@scheduler(")?;
    match &self.scheduler {
      Scheduler::LFP => f.write_str("\"lfp\"")?,
      Scheduler::AStar => f.write_str("\"a-star\"")?,
      Scheduler::DFS => f.write_str("\"dfs\"")?,
      Scheduler::Beam { beam_size } => f.write_fmt(format_args!("\"beam\", beam_size = {beam_size}"))?,
    }
    f.write_str(")")
  }
}

impl Display for Fact {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    if self.tag.is_some() {
      f.write_fmt(format_args!("{}::", self.tag))?;
    }
    f.write_fmt(format_args!("{}(", self.predicate))?;
    for (i, arg) in self.args.iter().enumerate() {
      f.write_fmt(format_args!("{:?}", arg))?;
      if i < self.args.len() - 1 {
        f.write_str(", ")?;
      }
    }
    f.write_str(")")
  }
}

impl Display for Rule {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_fmt(format_args!("{} :- {}", self.head, self.body))
  }
}

impl Display for Head {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    match self {
      Self::Atom(a) => a.fmt(f),
      Self::Disjunction(atoms) => {
        f.write_str("{")?;
        for (i, atom) in atoms.iter().enumerate() {
          if i > 0 {
            f.write_str("; ")?;
          }
          atom.fmt(f)?;
        }
        f.write_str("}")
      }
    }
  }
}

impl Display for Term {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    match self {
      Term::Constant(c) => c.fmt(f),
      Term::Variable(v) => v.fmt(f),
    }
  }
}

impl Display for Variable {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_str(&self.name)
  }
}

impl Display for Conjunction {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    for (i, arg) in self.args.iter().enumerate() {
      arg.fmt(f)?;
      if i < self.args.len() - 1 {
        f.write_str(", ")?;
      }
    }
    Ok(())
  }
}

impl Display for Literal {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    match self {
      Self::Atom(a) => a.fmt(f),
      Self::NegAtom(n) => n.fmt(f),
      Self::Assign(b) => b.fmt(f),
      Self::Constraint(c) => c.fmt(f),
      Self::Reduce(r) => r.fmt(f),
      Self::True => f.write_str("true"),
      Self::False => f.write_str("false"),
    }
  }
}

impl Display for Atom {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_fmt(format_args!("{}(", self.predicate))?;
    for (i, arg) in self.args.iter().enumerate() {
      arg.fmt(f)?;
      if i < self.args.len() - 1 {
        f.write_str(", ")?;
      }
    }
    f.write_str(")")
  }
}

impl Display for NegAtom {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_fmt(format_args!("~{}", self.atom))
  }
}

impl Display for Assign {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_fmt(format_args!("{} = {}", self.left.name, self.right))
  }
}

impl Display for AssignExpr {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    match self {
      Self::Binary(b) => b.fmt(f),
      Self::Unary(u) => u.fmt(f),
      Self::IfThenElse(i) => i.fmt(f),
      Self::Call(c) => c.fmt(f),
      Self::New(n) => n.fmt(f),
    }
  }
}

impl Display for BinaryAssignExpr {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_fmt(format_args!("{} {} {}", self.op1, self.op, self.op2))
  }
}

impl Display for UnaryAssignExpr {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    match &self.op {
      UnaryExprOp::Neg => f.write_fmt(format_args!("-{}", self.op1)),
      UnaryExprOp::Pos => f.write_fmt(format_args!("+{}", self.op1)),
      UnaryExprOp::Not => f.write_fmt(format_args!("!{}", self.op1)),
      UnaryExprOp::TypeCast(tgt) => f.write_fmt(format_args!("{} as {}", self.op1, tgt)),
    }
  }
}

impl Display for IfThenElseAssignExpr {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_fmt(format_args!(
      "if {} then {} else {}",
      self.cond, self.then_br, self.else_br
    ))
  }
}

impl Display for CallExpr {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_fmt(format_args!(
      "${}({})",
      self.function,
      self.args.iter().map(|a| format!("{}", a)).collect::<Vec<_>>().join("")
    ))
  }
}

impl Display for NewExpr {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_fmt(format_args!(
      "new {}({})",
      self.functor,
      self.args.iter().map(|a| format!("{}", a)).collect::<Vec<_>>().join("")
    ))
  }
}

impl Display for Constraint {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    match self {
      Self::Binary(b) => b.fmt(f),
      Self::Unary(u) => u.fmt(f),
    }
  }
}

impl Display for BinaryConstraintOp {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    match self {
      Self::Eq => f.write_str("=="),
      Self::Neq => f.write_str("!="),
      Self::Lt => f.write_str("<"),
      Self::Leq => f.write_str("<="),
      Self::Gt => f.write_str(">"),
      Self::Geq => f.write_str(">="),
    }
  }
}

impl Display for BinaryConstraint {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    f.write_fmt(format_args!("{} {} {}", self.op1, self.op, self.op2,))
  }
}

impl Display for UnaryConstraint {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    match self.op {
      UnaryConstraintOp::Not => f.write_fmt(format_args!("!{}", self.op1)),
    }
  }
}

impl Display for Reduce {
  fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
    if self.left_vars.len() > 1 {
      f.write_fmt(format_args!(
        "({}) := ",
        self
          .left_vars
          .iter()
          .map(|l| l.name.clone())
          .collect::<Vec<_>>()
          .join(", ")
      ))?;
    } else if self.left_vars.len() == 1 {
      f.write_fmt(format_args!("{} := ", self.left_vars[0].name))?;
    }
    f.write_str(&self.aggregator)?;
    if self.params.len() > 0 {
      let params = self.params.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", ");
      f.write_str(&params)?;
    }
    if self.has_exclamation_mark {
      f.write_str("!")?;
    }
    let group_by_vars = if self.group_by_vars.is_empty() {
      String::new()
    } else {
      let ids = self
        .group_by_vars
        .iter()
        .map(|i| i.name.clone())
        .collect::<Vec<_>>()
        .join(", ");
      format!("{{{}}}", ids)
    };
    let arg_vars = if self.arg_vars.is_empty() {
      String::new()
    } else {
      let ids = self
        .arg_vars
        .iter()
        .map(|i| i.name.clone())
        .collect::<Vec<_>>()
        .join(", ");
      format!("[{}]", ids)
    };
    let group_by_atom = if let Some(group_by_atom) = &self.group_by_formula {
      let group_by_vars = self
        .group_by_vars
        .iter()
        .map(|v| format!("{}", v))
        .collect::<Vec<_>>()
        .join(", ");
      format!(" where {}: {}", group_by_vars, group_by_atom)
    } else {
      format!("")
    };
    let to_agg_vars = self
      .to_aggregate_vars
      .iter()
      .map(|i| i.name.clone())
      .collect::<Vec<_>>()
      .join(", ");
    f.write_fmt(format_args!(
      "{}{}({}: {}{})",
      group_by_vars, arg_vars, to_agg_vars, self.body_formula, group_by_atom
    ))
  }
}
