use std::collections::*;

use super::super::*;

pub fn constant_prop(rule: &mut Rule) {
  let mut substitutions = HashMap::<_, _>::new();
  let mut ignore_literals = HashSet::new();
  let mut cannot_substitute = HashSet::new();
  let mut is_false = false;

  // Collect cannot substitute variables
  for literal in rule.body_literals() {
    match literal {
      Literal::Reduce(reduce) => {
        cannot_substitute.extend(reduce.variable_args().cloned());
      }
      _ => {}
    }
  }

  // Collect all substitutions
  for (i, literal) in rule.body_literals().enumerate() {
    match literal {
      Literal::Constraint(Constraint::Binary(b)) if &b.op == &BinaryConstraintOp::Eq => {
        match (&b.op1, &b.op2) {
          (Term::Variable(v), Term::Constant(c)) | (Term::Constant(c), Term::Variable(v)) => {
            // Make sure v is not among the `cannot_substitute`
            if cannot_substitute.contains(v) {
              continue;
            }

            // Ignore this literal
            ignore_literals.insert(i);

            // Substitute v with c
            if let Some(existing_c) = substitutions.get(v) {
              // Check if there is existing c that is equal to expected value
              if existing_c != c {
                // If not equal, then
                is_false = true;
              }
            } else {
              substitutions.insert(v.clone(), c.clone());
            }
          }
          _ => {}
        }
      }
      _ => {}
    }
  }

  // Closure for substitution
  let substitute_var = |v: &Variable| -> Term {
    if substitutions.contains_key(v) {
      Term::Constant(substitutions[v].clone())
    } else {
      Term::Variable(v.clone())
    }
  };
  let substitute_term = |t: &Term| -> Term {
    match t {
      Term::Constant(c) => Term::Constant(c.clone()),
      Term::Variable(v) => substitute_var(v),
    }
  };

  // If the rule is false, return a new rule with a false literal
  if is_false {
    // False body
    rule.body.args = vec![Literal::False];
  } else {
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
          Literal::Assign(a) => {
            if !substitutions.contains_key(&a.left) {
              Literal::Assign(Assign {
                left: a.left.clone(),
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
                },
              })
            } else {
              Literal::True
            }
          }
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
    let new_head = Head {
      predicate: rule.head.predicate.clone(),
      args: rule.head.args.iter().map(substitute_term).collect(),
    };

    // Update the rule
    rule.body.args = new_literals;
    rule.head = new_head;
  }
}
