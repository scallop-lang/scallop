use std::collections::*;

use super::*;

use crate::compiler::front::ast::*;
use crate::compiler::front::visitor::*;

#[derive(Clone, Debug)]
pub struct LocalBoundnessAnalysisContext<'a> {
  pub foreign_predicate_bindings: &'a ForeignPredicateBindings,
  pub expr_boundness: HashMap<Loc, bool>,
  pub dependencies: Vec<BoundnessDependency>,
  pub variable_locations: HashMap<String, Vec<Loc>>,
  pub constraints: Vec<Loc>,
  pub bounded_variables: BTreeSet<String>,
  pub errors: Vec<BoundnessAnalysisError>,
}

impl<'a> NodeVisitor for LocalBoundnessAnalysisContext<'a> {
  fn visit_atom(&mut self, atom: &Atom) {
    if let Some(binding) = self.foreign_predicate_bindings.get(atom.predicate()) {
      let bounded = atom.iter_arguments().enumerate().filter_map(|(i, a)| if binding[i].is_bound() { Some(a.location().clone()) } else { None } ).collect();
      let to_bound = atom.iter_arguments().enumerate().filter_map(|(i, a)| if binding[i].is_free() { Some(a.location().clone()) } else { None } ).collect();
      let dep = BoundnessDependency::ForeignPredicateArgs(bounded, to_bound);
      self.dependencies.push(dep);
    } else {
      for arg in atom.iter_arguments() {
        let loc = arg.location().clone();
        let dep = BoundnessDependency::RelationArg(loc);
        self.dependencies.push(dep);
      }
    }
  }

  fn visit_variable(&mut self, variable: &Variable) {
    // First put that into a location
    let name = variable.name().to_string();
    let loc = variable.location().clone();
    self.variable_locations.entry(name.clone()).or_default().push(loc);

    // If the variable is already bounded, we say the expression is bounded
    if self.bounded_variables.contains(&name) {
      self.expr_boundness.insert(variable.location().clone(), true);
    }
  }

  fn visit_constant(&mut self, constant: &Constant) {
    let dep = BoundnessDependency::Constant(constant.location().clone());
    self.dependencies.push(dep);
  }

  fn visit_constraint(&mut self, constraint: &Constraint) {
    // First put the constraint expression
    self.constraints.push(constraint.expr().location().clone());

    // Then put the equality if the constraint is a binary equality
    match constraint.expr() {
      Expr::Binary(b) => {
        if b.op().is_eq() {
          let dep = BoundnessDependency::ConstraintEquality(b.op1().location().clone(), b.op2().location().clone());
          self.dependencies.push(dep);
        }
      }
      _ => {}
    }
  }

  fn visit_binary_expr(&mut self, binary_expr: &BinaryExpr) {
    let op = binary_expr.op();
    let dep = if op.is_add_sub() {
      BoundnessDependency::AddSub(
        binary_expr.op1().location().clone(),
        binary_expr.op2().location().clone(),
        binary_expr.location().clone(),
      )
    } else {
      BoundnessDependency::BinaryOp(
        binary_expr.op1().location().clone(),
        binary_expr.op2().location().clone(),
        binary_expr.location().clone(),
      )
    };
    self.dependencies.push(dep);
  }

  fn visit_unary_expr(&mut self, unary_expr: &UnaryExpr) {
    let dep = BoundnessDependency::UnaryOp(unary_expr.op1().location().clone(), unary_expr.location().clone());
    self.dependencies.push(dep);
  }

  fn visit_if_then_else_expr(&mut self, ite_expr: &IfThenElseExpr) {
    let cl = ite_expr.cond().location().clone();
    let tl = ite_expr.then_br().location().clone();
    let el = ite_expr.else_br().location().clone();
    let dep = BoundnessDependency::IfThenElseOp(cl, tl, el, ite_expr.location().clone());
    self.dependencies.push(dep);
  }

  fn visit_call_expr(&mut self, call_expr: &CallExpr) {
    let arg_locs = call_expr.iter_args().map(|a| a.location().clone()).collect::<Vec<_>>();
    let dep = BoundnessDependency::CallOp(arg_locs, call_expr.location().clone());
    self.dependencies.push(dep);
  }
}

impl<'a> LocalBoundnessAnalysisContext<'a> {
  pub fn new(foreign_predicate_bindings: &'a ForeignPredicateBindings) -> Self {
    Self {
      foreign_predicate_bindings,
      expr_boundness: HashMap::new(),
      dependencies: Vec::new(),
      variable_locations: HashMap::new(),
      constraints: Vec::new(),
      bounded_variables: BTreeSet::new(),
      errors: Vec::new(),
    }
  }

  pub fn compute_boundness(&mut self) {
    // Local helper functions
    let update = |expr_boundness: &mut HashMap<Loc, bool>, l: &Loc, b: bool| {
      *expr_boundness.entry(l.clone()).or_default() |= b;
    };
    let get_mut =
      |expr_boundness: &mut HashMap<Loc, bool>, l: &Loc| -> bool { *expr_boundness.entry(l.clone()).or_default() };
    let get =
      |expr_boundness: &HashMap<Loc, bool>, l: &Loc| -> bool { expr_boundness.get(l).cloned().unwrap_or(false) };

    // Fixpoint iteration
    let mut old_expr_boundness = HashMap::new();
    let mut first_iteration = true;
    while first_iteration || old_expr_boundness != self.expr_boundness {
      first_iteration = false;
      old_expr_boundness = self.expr_boundness.clone();

      // First propagate the boundness through dependencies
      for dep in &self.dependencies {
        use BoundnessDependency::*;
        match dep {
          RelationArg(l) => {
            update(&mut self.expr_boundness, l, true);
          }
          ForeignPredicateArgs(bounded_args, to_bound_args) => {
            if bounded_args.iter().all(|l| get(&self.expr_boundness, l)) {
              for to_bound_arg in to_bound_args {
                update(&mut self.expr_boundness, to_bound_arg, true)
              }
            }
          }
          Constant(l) => {
            update(&mut self.expr_boundness, l, true);
          }
          BinaryOp(op1, op2, e) => {
            let b_op1 = get_mut(&mut self.expr_boundness, op1);
            let b_op2 = get_mut(&mut self.expr_boundness, op2);
            update(&mut self.expr_boundness, e, b_op1 && b_op2);
          }
          ConstraintEquality(op1, op2) => {
            let b_op1 = get_mut(&mut self.expr_boundness, op1);
            let b_op2 = get_mut(&mut self.expr_boundness, op2);
            if b_op1 && !b_op2 {
              update(&mut self.expr_boundness, op2, true);
            } else if !b_op1 && b_op2 {
              update(&mut self.expr_boundness, op1, true);
            }
          }
          AddSub(op1, op2, e) => {
            let b_op1 = get_mut(&mut self.expr_boundness, op1);
            let b_op2 = get_mut(&mut self.expr_boundness, op2);
            let b_e = get_mut(&mut self.expr_boundness, e);
            if b_op1 && b_op2 && !b_e {
              update(&mut self.expr_boundness, e, true);
            } else if b_op1 && !b_op2 && b_e {
              update(&mut self.expr_boundness, op2, true);
            } else if !b_op1 && b_op2 && b_e {
              update(&mut self.expr_boundness, op1, true);
            }
          }
          UnaryOp(op1, e) => {
            let b_op1 = get_mut(&mut self.expr_boundness, op1);
            update(&mut self.expr_boundness, e, b_op1);
          }
          IfThenElseOp(cond, then_br, else_br, e) => {
            let b_cond = get_mut(&mut self.expr_boundness, cond);
            let b_then_br = get_mut(&mut self.expr_boundness, then_br);
            let b_else_br = get_mut(&mut self.expr_boundness, else_br);
            update(&mut self.expr_boundness, e, b_cond && b_then_br && b_else_br);
          }
          CallOp(arg_locs, e) => {
            let mut b = true;
            for arg in arg_locs {
              b &= get_mut(&mut self.expr_boundness, arg);
            }
            update(&mut self.expr_boundness, e, b);
          }
        }
      }

      // Then propagate the boundness through variables
      for (_, locs) in &self.variable_locations {
        let var_bounded = locs.iter().any(|loc| get(&self.expr_boundness, loc));
        for loc in locs {
          update(&mut self.expr_boundness, loc, var_bounded);
        }
      }
    }

    // Check if all variables are bounded
    for (var, locs) in &self.variable_locations {
      let var_bounded = locs.iter().all(|loc| get(&self.expr_boundness, loc));
      if !var_bounded {
        self.errors.push(BoundnessAnalysisError::UnboundVariable {
          var_name: var.clone(),
          var_loc: locs[0].clone(),
        });
      } else {
        self.bounded_variables.insert(var.clone());
      }
    }

    // Check if all constraints are bounded
    for constraint in &self.constraints {
      if !get(&self.expr_boundness, constraint) {
        self.errors.push(BoundnessAnalysisError::ConstraintUnbound {
          loc: constraint.clone(),
        });
      }
    }
  }
}
