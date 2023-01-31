use std::collections::*;

use super::super::*;

pub fn constant_prop(rule: &mut Rule) {
  let mut substitutions = HashMap::<_, _>::new();
  let mut ignore_literals = HashSet::new();
  let mut cannot_substitute = HashSet::new();
  let mut is_false = false;

  // Collect cannot substitute variables
  let mut visited_literal_ids = HashSet::new();
  loop {
    let prev_cannot_substitute_size = cannot_substitute.len();

    // Iterate through all the literals
    for (id, literal) in rule.body_literals().enumerate() {
      // Skip visited literals
      if visited_literal_ids.contains(&id) {
        continue;
      }

      // Extend the cannot substitute variables
      match literal {
        Literal::Reduce(reduce) => {
          visited_literal_ids.insert(id);
          cannot_substitute.extend(reduce.variable_args().cloned());
        }
        Literal::Assign(assign) => {
          for arg in assign.variable_args() {
            if cannot_substitute.contains(arg) {
              visited_literal_ids.insert(id);
              cannot_substitute.insert(assign.left.clone());
            }
          }
        }
        _ => {}
      }
    }

    // Exit the loop if there is no more new variable
    if prev_cannot_substitute_size >= cannot_substitute.len() {
      break;
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
                  AssignExpr::Call(c) => AssignExpr::Call(CallExpr {
                    function: c.function.clone(),
                    args: c.args.iter().map(substitute_term).collect(),
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
