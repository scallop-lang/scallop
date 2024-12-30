use std::collections::*;

use crate::common::foreign_predicate::*;
use crate::common::value_type::*;

use super::*;

/// The context for constructing a query plan
#[derive(Clone, Debug)]
pub struct QueryPlanContext<'a> {
  pub head_vars: HashSet<Variable>,
  pub reduces: Vec<Reduce>,
  pub pos_atoms: Vec<Atom>,
  pub neg_atoms: Vec<Atom>,
  pub assigns: Vec<Assign>,
  pub constraints: Vec<Constraint>,
  pub foreign_predicate_pos_atoms: Vec<Atom>,
  pub stratum: &'a Stratum,
  pub foreign_predicate_registry: &'a ForeignPredicateRegistry,
}

impl<'a> QueryPlanContext<'a> {
  /// Create a new query plan context from a rule
  pub fn from_rule(
    stratum: &'a Stratum,
    foreign_predicate_registry: &'a ForeignPredicateRegistry,
    rule: &Rule,
  ) -> Self {
    // First create an empty context
    let mut ctx = Self {
      head_vars: rule.head.variable_args().into_iter().cloned().collect(),
      reduces: vec![],
      pos_atoms: vec![],
      neg_atoms: vec![],
      assigns: vec![],
      constraints: vec![],
      foreign_predicate_pos_atoms: vec![],
      stratum,
      foreign_predicate_registry,
    };

    // Then fill it with the literals extracted from the rule
    for literal in rule.body_literals() {
      match literal {
        Literal::Reduce(r) => ctx.reduces.push(r.clone()),
        Literal::Atom(a) => {
          if foreign_predicate_registry.contains(&a.predicate) {
            ctx.foreign_predicate_pos_atoms.push(a.clone());
          } else {
            ctx.pos_atoms.push(a.clone())
          }
        }
        Literal::NegAtom(n) => ctx.neg_atoms.push(n.atom.clone()),
        Literal::Assign(a) => ctx.assigns.push(a.clone()),
        Literal::Constraint(c) => ctx.constraints.push(c.clone()),
        Literal::True => {}
        Literal::False => {
          panic!("[Internal Error] Rule with false literal not removed")
        }
      }
    }
    ctx
  }

  /// Find all the bounded arguments given the set of positive atoms
  pub fn bounded_args_from_pos_atoms_set(&self, set: &Vec<&usize>, include_new: bool) -> HashSet<Variable> {
    let mut base_bounded_args = HashSet::new();

    // Add the base cases: all the arguments in the positive atoms form the base bounded args
    for atom in self
      .pos_atoms
      .iter()
      .enumerate()
      .filter_map(|(i, m)| if set.contains(&&i) { Some(m) } else { None })
    {
      for var in atom.args.iter().filter_map(|a| match a {
        Term::Variable(a) => Some(a),
        _ => None,
      }) {
        base_bounded_args.insert(var.clone());
      }
    }

    // Fix point iteration
    loop {
      let mut new_bounded_args = base_bounded_args.clone();

      // Find the bounded args from the assigns
      for assign in &self.assigns {
        let can_bound = match &assign.right {
          AssignExpr::Binary(b) => {
            let op1_bounded = term_is_bounded(&new_bounded_args, &b.op1);
            let op2_bounded = term_is_bounded(&new_bounded_args, &b.op1);
            op1_bounded && op2_bounded
          }
          AssignExpr::Unary(u) => term_is_bounded(&new_bounded_args, &u.op1),
          AssignExpr::IfThenElse(i) => {
            let cond_bounded = term_is_bounded(&new_bounded_args, &i.cond);
            let then_br_bounded = term_is_bounded(&new_bounded_args, &i.then_br);
            let else_br_bounded = term_is_bounded(&new_bounded_args, &i.else_br);
            cond_bounded && then_br_bounded && else_br_bounded
          }
          AssignExpr::Call(c) => c.args.iter().all(|a| term_is_bounded(&new_bounded_args, a)),
          AssignExpr::New(n) => {
            // If not include new, we do not bound the left variable
            include_new && n.args.iter().all(|a| term_is_bounded(&new_bounded_args, a))
          }
        };
        if can_bound {
          new_bounded_args.insert(assign.left.clone());
        }
      }

      // Find the bounded args from the foreign predicate atoms
      for atom in &self.foreign_predicate_pos_atoms {
        // First find the predicate from the registry
        let predicate = self.foreign_predicate_registry.get(&atom.predicate).unwrap();

        // Then check if all the to-bound arguments are bounded
        let can_bound = atom
          .args
          .iter()
          .take(predicate.num_bounded())
          .all(|a| term_is_bounded(&new_bounded_args, a));

        // If it can be bounded, add the rest of the arguments to the bounded args
        if can_bound {
          for arg in atom.args.iter().skip(predicate.num_bounded()) {
            if let Term::Variable(v) = arg {
              new_bounded_args.insert(v.clone());
            }
          }
        }
      }

      // Check if the fix point is reached
      if new_bounded_args == base_bounded_args {
        break new_bounded_args;
      } else {
        base_bounded_args = new_bounded_args;
      }
    }
  }

  fn pos_atom_arcs(&self, beam_size: usize) -> State {
    let atom_relations = self.pos_atoms.iter().enumerate().map(|(i, atom)| (i, atom.predicate.clone())).collect();

    // If there is no positive atom, return an empty state
    if self.pos_atoms.is_empty() {
      return State::new(atom_relations);
    }

    // Maintain a priority queue of searching states
    let mut priority_queue = BinaryHeap::new();
    priority_queue.push(State::new(atom_relations));

    // Maintain a set of final states
    let mut final_states = BinaryHeap::new();

    // Start the (beam) search process
    while !priority_queue.is_empty() {
      let mut temp_queue = BinaryHeap::new();

      // Find the next states and push them into the queue
      while !priority_queue.is_empty() {
        let top = priority_queue.pop().unwrap();
        for next_state in top.next_states(self, beam_size) {
          if next_state.bounded_all(self) {
            final_states.push(next_state);
          } else {
            temp_queue.push(next_state);
          }
        }
      }

      // Retain only top `beam_size` inside of the priority_queue
      for _ in 0..beam_size {
        if let Some(elem) = temp_queue.pop() {
          priority_queue.push(elem);
        } else {
          break;
        }
      }
    }

    // At the end, use the best state in `final_states`
    final_states.pop().unwrap()
  }

  fn try_apply_constraint(&self, applied_constraints: &mut HashSet<usize>, fringe: Plan) -> Plan {
    // Apply as many constraints as possible
    let node = fringe;
    let mut to_apply_constraints = vec![];
    for (i, constraint) in self.constraints.iter().enumerate() {
      if !applied_constraints.contains(&i) {
        let can_apply = constraint.variable_args().iter().all(|v| node.bounded_vars.contains(v));
        if can_apply {
          applied_constraints.insert(i);
          to_apply_constraints.push(constraint.clone());
        }
      }
    }
    if to_apply_constraints.is_empty() {
      node
    } else {
      let new_bounded_vars = node.bounded_vars.clone();

      Plan {
        bounded_vars: new_bounded_vars,
        ram_node: HighRamNode::Filter(Box::new(node), to_apply_constraints),
      }
    }
  }

  /// Try applying as many assigns as possible
  fn try_apply_non_new_assigns(&self, applied_assigns: &mut HashSet<usize>, mut fringe: Plan) -> Plan {
    // Find all the assigns that are needed to bound the need_projection_vars
    let mut bounded_vars = fringe.bounded_vars.clone();
    loop {
      let mut new_projections = Vec::new();

      // Check if we can apply more assigns
      for (i, assign) in self.assigns.iter().enumerate() {
        if !applied_assigns.contains(&i)
          && !bounded_vars.contains(&assign.left)
          && !assign.right.is_new_expr()
          && assign.variable_args().into_iter().all(|v| bounded_vars.contains(v))
        {
          applied_assigns.insert(i);
          new_projections.push(assign.clone());
        }
      }

      // Create projected left node
      if new_projections.is_empty() {
        break fringe;
      } else {
        bounded_vars.extend(new_projections.iter().map(|i| i.left.clone()));
        fringe = Plan {
          bounded_vars: bounded_vars.clone(),
          ram_node: HighRamNode::Project(Box::new(fringe), new_projections),
        };
      }
    }
  }

  /// Get the essential information for analyzing a foreign predicate atom
  fn foreign_predicate_atom_info<'b, 'c>(
    &'b self,
    atom: &'c Atom,
  ) -> (
    &'b DynamicForeignPredicate,
    Vec<(usize, &'c Term)>,
    Vec<(usize, &'c Term)>,
  ) {
    let pred = self.foreign_predicate_registry.get(&atom.predicate).unwrap();
    let (to_bound_arguments, free_arguments): (Vec<_>, Vec<_>) =
      atom.args.iter().enumerate().partition(|(i, _)| *i < pred.num_bounded());
    (pred, to_bound_arguments, free_arguments)
  }

  fn foreign_predicate_constant_constraints(&self, atom: &Atom, arguments: &Vec<(usize, &Term)>) -> Vec<Constraint> {
    arguments
      .iter()
      .filter_map(|(i, a)| match a {
        Term::Constant(c) => {
          let op1 = Term::variable(format!("c#{}#{}", &atom.predicate, i), ValueType::type_of(c));
          let op2 = Term::Constant(c.clone());
          Some(Constraint::eq(op1, op2))
        }
        Term::Variable(_) => None,
      })
      .collect()
  }

  fn foreign_predicate_equality_constraints(&self, var_eq: &Vec<(Variable, Variable)>) -> Vec<Constraint> {
    var_eq
      .iter()
      .map(|(v1, v2)| Constraint::eq(Term::Variable(v1.clone()), Term::Variable(v2.clone())))
      .collect()
  }

  /// Given a list of free arguments, rename them to avoid name conflicts
  ///
  /// Specifically, the renaming is done in the following way:
  ///
  /// - If an argument is a variable
  ///   - If the variable only occurs once, then we do not rename it
  ///   - If the variable occurs more than once, then we rename it to `var#i` where `i` is the number of occurrences
  /// - If an argument is a constant, then we rename it to `c#predicate#i` where `i` is the position of the argument
  ///
  /// In this case, all the arguments become distinct variables, where original variable names are preserved
  fn rename_free_arguments(
    &self,
    predicate: &str,
    arguments: &Vec<(usize, &Term)>,
    occurred: &HashSet<Variable>,
  ) -> (Vec<Variable>, Vec<(Variable, Variable)>) {
    // First build a map from variable names to the number of occurrences
    let mut var_occurrences: HashMap<String, (Variable, usize)> =
      occurred.iter().map(|v| (v.name.clone(), (v.clone(), 0))).collect();
    let mut var_equivalences: Vec<(Variable, Variable)> = Vec::new();
    let vars = arguments
      .iter()
      .map(|(i, a)| match a {
        Term::Variable(v) => {
          if let Some((old_var, occ)) = var_occurrences.get_mut(&v.name) {
            // Create a new variable different than the previous argument
            let new_var = Variable::new(format!("{}#{}", v.name, occ), v.ty.clone());

            // Add to the list of equivalences
            var_equivalences.push((old_var.clone(), new_var.clone()));

            // Update the information stored in the variable occurrences
            *occ += 1;
            *old_var = new_var.clone();

            // Return the new variable
            new_var
          } else {
            // Insert into variable occurrences
            var_occurrences.insert(v.name.clone(), (v.clone(), 1));

            // Return the variable
            v.clone()
          }
        }
        Term::Constant(c) => Variable::new(format!("c#{}#{}", predicate, i), ValueType::type_of(c)),
      })
      .collect::<Vec<_>>();
    (vars, var_equivalences)
  }

  ///
  fn compute_foreign_predicate_ground_atom(
    &self,
    atom: &Atom,
    pred: &DynamicForeignPredicate,
    free_arguments: &Vec<(usize, &Term)>,
  ) -> Plan {
    // The atom is grounded
    let (all_vars, var_eq) = self.rename_free_arguments(&atom.predicate, free_arguments, &HashSet::new());

    // Create a ground plan
    let args = atom
      .args
      .iter()
      .take(pred.num_bounded())
      .cloned()
      .chain(all_vars.iter().cloned().map(Term::Variable))
      .collect();
    let ground_atom = Atom::new(atom.predicate.clone(), args);
    let ground_plan = Plan {
      bounded_vars: all_vars.iter().cloned().collect(),
      ram_node: HighRamNode::ForeignPredicateGround(ground_atom),
    };

    // Get all the constraints
    let const_constraints = self.foreign_predicate_constant_constraints(atom, &free_arguments);
    let eq_constraints = self.foreign_predicate_equality_constraints(&var_eq);
    let constraints = vec![const_constraints, eq_constraints].concat();

    // Create a plan;
    // if there are constraints, we need to create a filter plan on top of the ground plan
    // Otherwise, we can just return the ground plan
    if !constraints.is_empty() {
      // Find the bounded vars
      // Note that we do not use all the variables occurring in the constraints
      let new_bounded_vars = free_arguments
        .iter()
        .filter_map(|(_, a)| match a {
          Term::Variable(v) => Some(v.clone()),
          _ => None,
        })
        .collect();

      // Create a plan with filters on the constants
      let filter_plan = Plan {
        bounded_vars: new_bounded_vars,
        ram_node: HighRamNode::filter(ground_plan, constraints),
      };

      filter_plan
    } else {
      ground_plan
    }
  }

  fn compute_foreign_predicate_join_atom(
    &self,
    left: Plan,
    atom: &Atom,
    pred: &DynamicForeignPredicate,
    free_arguments: &Vec<(usize, &Term)>,
  ) -> Plan {
    // Get the arguments to the foreign predicate
    let occurred_variables = left.bounded_vars.iter().cloned().collect();
    let (free_vars, var_eq) = self.rename_free_arguments(&atom.predicate, free_arguments, &occurred_variables);

    // Create an atom
    let args = atom
      .args
      .iter()
      .take(pred.num_bounded())
      .cloned()
      .chain(free_vars.iter().cloned().map(Term::Variable))
      .collect();
    let to_join_atom = Atom::new(atom.predicate.clone(), args);

    // Create the join plan
    let join_plan = Plan {
      bounded_vars: left
        .bounded_vars
        .iter()
        .cloned()
        .chain(free_vars.iter().cloned())
        .collect(),
      ram_node: HighRamNode::foreign_predicate_join(left.clone(), to_join_atom),
    };

    // Get all the constraints
    let const_constraints = self.foreign_predicate_constant_constraints(atom, &free_arguments);
    let eq_constraints = self.foreign_predicate_equality_constraints(&var_eq);
    let constraints = vec![const_constraints, eq_constraints].concat();

    // Create a plan;
    // if there are constraints, we need to create a filter plan on top of the ground plan
    // Otherwise, we can just return the ground plan
    if !constraints.is_empty() {
      // Find the bounded vars
      // Note that we do not use all the variables occurring in the constraints
      let right_bounded_vars: Vec<_> = free_arguments
        .iter()
        .filter_map(|(_, a)| match a {
          Term::Variable(v) => Some(v.clone()),
          _ => None,
        })
        .collect();

      // Create a plan with filters on the constants
      let filter_plan = Plan {
        bounded_vars: left
          .bounded_vars
          .iter()
          .chain(right_bounded_vars.iter())
          .cloned()
          .collect(),
        ram_node: HighRamNode::filter(join_plan, constraints),
      };

      filter_plan
    } else {
      join_plan
    }
  }

  /// Try to apply the foreign predicate atoms in the context.
  /// Will scan all the foreign predicate atoms and see if there are any that can be applied.
  ///
  /// - `applied_foreign_predicate_atoms`: The set of already applied foreign predicate atoms, represented by their index
  /// - `fringe`: The current Plan to apply the foreign predicate atoms
  fn try_apply_foreign_predicate_atom(
    &self,
    applied_foreign_predicate_atoms: &mut HashSet<usize>,
    mut fringe: Plan,
  ) -> Plan {
    // Find all the foreign predicate atoms
    let bounded_vars = fringe.bounded_vars.clone();
    loop {
      let mut applied = false;

      // Check if we can apply more foreign predicate atoms
      for (i, atom) in self.foreign_predicate_pos_atoms.iter().enumerate() {
        if !applied_foreign_predicate_atoms.contains(&i) {
          // Get the foreign predicate information
          let (pred, to_bound_arguments, free_arguments) = self.foreign_predicate_atom_info(atom);

          // Check if all the to-bound arguments are bounded; if so, it means that we can apply the atom
          if to_bound_arguments
            .iter()
            .all(|(_, a)| term_is_bounded(&bounded_vars, a))
          {
            // Mark the atom as applied
            applied_foreign_predicate_atoms.insert(i);

            // There are 3 kinds of foreign predicate atoms:
            // 1. There are no free arguments (i.e. all arguments are bounded)
            // 2. Ground atom (i.e. all bounded arguments are constants)
            // 3. Joining atom (i.e. some bounded arguments are variables)
            // For each of these cases, we need to create a different plan
            if free_arguments.is_empty() {
              // The atom is completely bounded; we add a foreign predicate constraint plan
              fringe = Plan {
                bounded_vars: bounded_vars.clone(),
                ram_node: HighRamNode::ForeignPredicateConstraint(Box::new(fringe), atom.clone()),
              };
            } else if to_bound_arguments.iter().all(|(_, a)| a.is_constant()) {
              let plan = self.compute_foreign_predicate_ground_atom(atom, pred, &free_arguments);

              // Connect it with the existing plan
              fringe = Plan {
                bounded_vars: fringe.bounded_vars.union(&plan.bounded_vars).cloned().collect(),
                ram_node: HighRamNode::join(fringe, plan),
              };
            } else {
              // The atom is bounded and new values can be generated
              fringe = self.compute_foreign_predicate_join_atom(fringe, atom, pred, &free_arguments);
            }

            // Found an atom that can be applied
            applied = true;
          }
        }
      }

      // Break the loop if no more foreign predicate atoms can be applied
      if !applied {
        break fringe;
      }
    }
  }

  fn try_apply(
    &self,
    mut fringe: Plan,
    applied_assigns: &mut HashSet<usize>,
    applied_constraints: &mut HashSet<usize>,
    applied_foreign_predicates: &mut HashSet<usize>,
  ) -> Plan {
    // Note: We always apply constraint first and then assigns
    let mut num_applied_assigns = applied_assigns.len();
    let mut num_applied_constraints = applied_constraints.len();
    let mut num_applied_fp = applied_foreign_predicates.len();
    loop {
      fringe = self.try_apply_non_new_assigns(applied_assigns, fringe);
      fringe = self.try_apply_constraint(applied_constraints, fringe);
      fringe = self.try_apply_foreign_predicate_atom(applied_foreign_predicates, fringe);

      // Check if anything is applied
      let does_apply_assign = applied_assigns.len() != num_applied_assigns;
      let does_apply_constraint = applied_constraints.len() != num_applied_constraints;
      let does_apply_fp = applied_foreign_predicates.len() != num_applied_fp;

      // If so, we continue in a loop
      if does_apply_assign || does_apply_constraint || does_apply_fp {
        num_applied_assigns = applied_assigns.len();
        num_applied_constraints = applied_constraints.len();
        num_applied_fp = applied_foreign_predicates.len();
      } else {
        break fringe;
      }
    }
  }

  fn is_ground_foreign_atom(&self, atom: &Atom) -> bool {
    let pred = self.foreign_predicate_registry.get(&atom.predicate).unwrap();
    atom.args.iter().take(pred.num_bounded()).all(|a| a.is_constant())
  }

  /// The main entry function that computes a query plan from a sequence of arcs
  fn get_query_plan(&self, arcs: &Vec<Arc>) -> Plan {
    // ==== Stage 1: Helper Functions (Closures) ====

    // Store the applied constraints
    let mut applied_constraints = HashSet::new();
    let mut applied_assigns = HashSet::new();
    let mut applied_foreign_predicates = HashSet::new();

    // ==== Stage 2: Building the RAM tree bottom-up, starting with reduces ====

    // Build the first fringe
    let (mut fringe, start_arc_id) = if self.reduces.is_empty() {
      // There is no reduce
      if arcs.is_empty() {
        // There is no arc
        if self.foreign_predicate_pos_atoms.is_empty() {
          // There is no reduce and there is no arc and there is no foreign predicate atom
          let node = Plan::unit();
          let node = self.try_apply(
            node,
            &mut applied_assigns,
            &mut applied_constraints,
            &mut applied_foreign_predicates,
          );
          (node, 0)
        } else {
          // Find the foreign predicate atom
          if let Some((i, atom)) = self
            .foreign_predicate_pos_atoms
            .iter()
            .enumerate()
            .find(|(_, a)| self.is_ground_foreign_atom(a))
          {
            applied_foreign_predicates.insert(i); // Mark the atom as applied
            let (pred, _, free_arguments) = self.foreign_predicate_atom_info(atom);
            let plan = self.compute_foreign_predicate_ground_atom(atom, pred, &free_arguments);
            let plan = self.try_apply(
              plan,
              &mut applied_assigns,
              &mut applied_constraints,
              &mut applied_foreign_predicates,
            );
            (plan, 0)
          } else {
            panic!("[Internal Error] No foreign predicate atom is ground; should not happen");
          }
        }
      } else {
        // If there is no reduce, find the first arc
        let first_arc = &arcs[0];
        let node = Plan {
          bounded_vars: self.pos_atoms[first_arc.right].variable_args().cloned().collect(),
          ram_node: HighRamNode::Ground(self.pos_atoms[first_arc.right].clone()),
        };

        // Note: We always apply constraint first and then assigns
        let node = self.try_apply(
          node,
          &mut applied_assigns,
          &mut applied_constraints,
          &mut applied_foreign_predicates,
        );
        (node, 1)
      }
    } else {
      // If there is reduce, create a joined reduce
      let first_reduce = &self.reduces[0];
      let mut node = Plan {
        bounded_vars: first_reduce.variable_args().cloned().collect(),
        ram_node: HighRamNode::Reduce(first_reduce.clone()),
      };

      // Get more reduces
      for reduce in &self.reduces[1..] {
        let left = node;
        let right_bounded_vars = reduce.variable_args().cloned().collect::<HashSet<_>>();
        let right = Plan {
          bounded_vars: right_bounded_vars.clone(),
          ram_node: HighRamNode::Reduce(reduce.clone()),
        };
        node = Plan {
          bounded_vars: left.bounded_vars.union(&right_bounded_vars).cloned().collect(),
          ram_node: HighRamNode::join(left, right),
        };
      }

      // Note: We always apply constraint first and then assigns
      let node = self.try_apply(
        node,
        &mut applied_assigns,
        &mut applied_constraints,
        &mut applied_foreign_predicates,
      );
      (node, 0)
    };

    // ==== Stage 3. Iterate through all the arcs, build the tree from bottom-up ====
    for arc in &arcs[start_arc_id..] {
      // Build the simple tree
      if arc.left.is_empty() {
        // A node that is not related to any of the node before; need product
        let left = fringe;
        let right = Plan {
          bounded_vars: self.pos_atoms[arc.right].variable_args().cloned().collect(),
          ram_node: HighRamNode::Ground(self.pos_atoms[arc.right].clone()),
        };
        let new_bounded_vars = left.bounded_vars.union(&right.bounded_vars).cloned().collect();
        let new_ram_node = HighRamNode::join(left, right);
        fringe = Plan {
          bounded_vars: new_bounded_vars,
          ram_node: new_ram_node,
        };
      } else {
        let left = fringe;

        // Create right node
        let right_bounded_vars = self.pos_atoms[arc.right]
          .variable_args()
          .cloned()
          .collect::<HashSet<_>>();
        let right = Plan {
          bounded_vars: right_bounded_vars,
          ram_node: HighRamNode::Ground(self.pos_atoms[arc.right].clone()),
        };

        // Create joined node
        fringe = Plan {
          bounded_vars: left.bounded_vars.union(&right.bounded_vars).cloned().collect(),
          ram_node: HighRamNode::join(left, right),
        };
      }

      // Apply all the things
      fringe = self.try_apply(
        fringe,
        &mut applied_assigns,
        &mut applied_constraints,
        &mut applied_foreign_predicates,
      );
    }

    // ==== Stage 4: Apply negative atoms ====
    for neg_atom in &self.neg_atoms {
      let neg_node = Plan {
        bounded_vars: neg_atom.variable_args().cloned().collect(),
        ram_node: HighRamNode::Ground(neg_atom.clone()),
      };
      fringe = Plan {
        bounded_vars: fringe.bounded_vars.clone(),
        ram_node: HighRamNode::antijoin(fringe, neg_node),
      };
    }

    // ==== Stage 5: Apply new entity assigns ====
    let new_entity_assigns = self
      .assigns
      .iter()
      .filter(|a| a.right.is_new_expr())
      .cloned()
      .collect::<Vec<_>>();
    if new_entity_assigns.len() > 0 {
      let all_bounded_variables = new_entity_assigns
        .iter()
        .map(|a| &a.left)
        .chain(fringe.bounded_vars.iter())
        .cloned()
        .collect();
      fringe = Plan {
        bounded_vars: all_bounded_variables,
        ram_node: HighRamNode::project(fringe, new_entity_assigns),
      };
    }

    fringe
  }

  pub fn query_plan(&self) -> Plan {
    let beam_size = 5;
    let state = self.pos_atom_arcs(beam_size);
    self.get_query_plan(&state.arcs)
  }
}

fn term_is_bounded(bounded_vars: &HashSet<Variable>, term: &Term) -> bool {
  match term {
    Term::Variable(v) => bounded_vars.contains(v),
    _ => true,
  }
}
