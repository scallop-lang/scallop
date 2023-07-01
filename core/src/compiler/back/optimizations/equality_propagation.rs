use std::collections::*;

use super::super::*;

pub fn propagate_equality(rule: &mut Rule) {
  let mut substitutions = HashMap::<_, Variable>::new();
  let mut ignore_literals = HashSet::new();
  let mut cannot_substitute = HashSet::<Variable>::new();

  // Collect cannot substitute variables
  for literal in rule.body_literals() {
    match literal {
      Literal::Reduce(r) => {
        cannot_substitute.extend(r.variable_args().cloned());
      }
      _ => {}
    }
  }

  // Find all the bounded variables by atom and assign
  let bounded = bounded_by_atom_and_assign(rule);

  // Collect all substitutions
  for (i, literal) in rule.body_literals().enumerate() {
    match literal {
      Literal::Constraint(Constraint::Binary(b)) if &b.op == &BinaryConstraintOp::Eq => {
        match (&b.op1, &b.op2) {
          (Term::Variable(v1), Term::Variable(v2)) => {
            // Make sure non of these are among the "cannot substitute" variables
            if cannot_substitute.contains(v1) || cannot_substitute.contains(v2) {
              continue;
            } else if bounded.contains(v1) && bounded.contains(v2) {
              // If both sides are derivable from atom/assign
              continue;
            }

            // Add this literal into the ignore literals
            ignore_literals.insert(i);

            // Get a substitution
            if substitutions.contains_key(v1) {
              substitutions.insert(v2.clone(), substitutions[v1].clone());
            } else if substitutions.contains_key(v2) {
              substitutions.insert(v1.clone(), substitutions[v2].clone());
            } else {
              substitutions.insert(v1.clone(), v2.clone());
            }
          }
          _ => {}
        }
      }
      _ => {}
    }
  }

  // Closure for substitution
  let substitute_var = |v: &Variable| -> Variable {
    if substitutions.contains_key(v) {
      substitutions[v].clone()
    } else {
      v.clone()
    }
  };
  let substitute_term = |t: &Term| -> Term {
    match t {
      Term::Constant(c) => Term::Constant(c.clone()),
      Term::Variable(v) => Term::Variable(substitute_var(v)),
    }
  };

  // Apply substitutions for each literal
  let mut new_literals = vec![];
  for (i, literal) in rule.body_literals().enumerate() {
    if !ignore_literals.contains(&i) {
      // Perform substitution on this variable
      let new_literal = match literal {
        Literal::Atom(a) => Literal::Atom(Atom {
          predicate: a.predicate.clone(),
          args: a.args.iter().map(substitute_term).collect(),
        }),
        Literal::Assign(a) => Literal::Assign(Assign {
          left: substitute_var(&a.left),
          right: match &a.right {
            AssignExpr::Binary(b) => AssignExpr::Binary(BinaryAssignExpr {
              op: b.op.clone(),
              op1: substitute_term(&b.op1),
              op2: substitute_term(&b.op2),
            }),
            AssignExpr::Unary(u) => AssignExpr::Unary(UnaryAssignExpr {
              op: u.op.clone(),
              op1: substitute_term(&u.op1),
            }),
            AssignExpr::IfThenElse(i) => AssignExpr::IfThenElse(IfThenElseAssignExpr {
              cond: substitute_term(&i.cond),
              then_br: substitute_term(&i.then_br),
              else_br: substitute_term(&i.else_br),
            }),
            AssignExpr::Call(c) => AssignExpr::Call(CallExpr {
              function: c.function.clone(),
              args: c.args.iter().map(substitute_term).collect(),
            }),
            AssignExpr::New(n) => AssignExpr::New(NewExpr {
              functor: n.functor.clone(),
              args: n.args.iter().map(substitute_term).collect(),
            }),
          },
        }),
        Literal::Constraint(Constraint::Binary(b)) => Literal::Constraint(Constraint::Binary(BinaryConstraint {
          op: b.op.clone(),
          op1: substitute_term(&b.op1),
          op2: substitute_term(&b.op2),
        })),
        Literal::Constraint(Constraint::Unary(u)) => Literal::Constraint(Constraint::Unary(UnaryConstraint {
          op: u.op.clone(),
          op1: substitute_term(&u.op1),
        })),
        Literal::NegAtom(n) => Literal::NegAtom(NegAtom {
          atom: Atom {
            predicate: n.atom.predicate.clone(),
            args: n.atom.args.iter().map(substitute_term).collect(),
          },
        }),
        _ => literal.clone(),
      };

      // Add this new literal
      new_literals.push(new_literal);
    }
  }

  // Apply substitution to the head
  let new_head = rule.head.substitute(substitute_term);

  // Update the rule into this new rule
  *rule = Rule {
    attributes: rule.attributes.clone(),
    head: new_head,
    body: Conjunction { args: new_literals },
  }
}

fn bounded_by_atom_and_assign(rule: &Rule) -> HashSet<Variable> {
  let mut bounded = rule
    .body_literals()
    .flat_map(|l| match l {
      Literal::Atom(a) => a.variable_args().cloned().collect::<Vec<_>>(),
      _ => vec![],
    })
    .collect::<HashSet<_>>();
  loop {
    let old_bounded = bounded.clone();
    for lit in rule.body_literals() {
      match lit {
        Literal::Assign(a) => {
          if !bounded.contains(&a.left) {
            if a.variable_args().iter().all(|arg| bounded.contains(arg)) {
              bounded.insert(a.left.clone());
            }
          }
        }
        _ => {}
      }
    }
    if old_bounded == bounded {
      break bounded;
    }
  }
}
