use std::collections::*;

use super::*;
use crate::common::value_type::*;
use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub struct LocalTypeInferenceContext {
  pub rule_loc: Loc,
  pub atom_arities: HashMap<String, Vec<(usize, Loc)>>,
  pub unifications: Vec<Unification>,
  pub vars_of_same_type: Vec<(String, String)>,
  pub var_types: HashMap<String, (TypeSet, Loc)>,
  pub constraints: Vec<Loc>,
  pub errors: Vec<TypeInferenceError>,
}

impl LocalTypeInferenceContext {
  pub fn new(rule_loc: Loc) -> Self {
    Self {
      rule_loc,
      atom_arities: HashMap::new(),
      unifications: Vec::new(),
      vars_of_same_type: Vec::new(),
      var_types: HashMap::new(),
      constraints: Vec::new(),
      errors: Vec::new(),
    }
  }

  pub fn from_rule(r: &Rule) -> Self {
    let mut ctx = Self::new(r.location().clone());
    ctx.walk_rule(r);
    ctx
  }

  pub fn from_atom(a: &Atom) -> Self {
    let mut ctx = Self::new(a.location().clone());
    ctx.walk_atom(a);
    ctx
  }

  pub fn unify_atom_arities(
    &self,
    inferred_relation_types: &mut HashMap<String, (Vec<TypeSet>, Loc)>,
  ) -> Result<(), TypeInferenceError> {
    for (pred, arities) in &self.atom_arities {
      // Make sure we have inferred relation types for the predicate
      if !inferred_relation_types.contains_key(pred) {
        let (arity, atom_loc) = &arities[0];
        let init_types = vec![TypeSet::Any(Loc::default()); *arity];
        inferred_relation_types.insert(pred.clone(), (init_types, atom_loc.clone()));
      }

      // Make sure the arity matches
      let (tys, source_loc) = &inferred_relation_types[pred];
      for (arity, atom_loc) in arities {
        if arity != &tys.len() {
          return Err(TypeInferenceError::ArityMismatch {
            predicate: pred.clone(),
            expected: tys.len(),
            actual: *arity,
            source_loc: source_loc.clone(),
            mismatch_loc: atom_loc.clone(),
          });
        }
      }
    }
    Ok(())
  }

  pub fn populate_inference_data(
    &self,
    inferred_relation_expr: &mut HashMap<(String, usize), BTreeSet<Loc>>,
    inferred_var_expr: &mut HashMap<Loc, HashMap<String, BTreeSet<Loc>>>,
    inferred_expr_types: &mut HashMap<Loc, TypeSet>,
  ) {
    for unif in &self.unifications {
      match unif {
        Unification::IthArgOfRelation(e, p, i) => {
          inferred_relation_expr
            .entry((p.clone(), *i))
            .or_default()
            .insert(e.clone());
        }
        Unification::OfVariable(e, v) => {
          inferred_var_expr
            .entry(self.rule_loc.clone())
            .or_default()
            .entry(v.clone())
            .or_default()
            .insert(e.clone());
        }
        _ => {}
      }
    }

    for (var, (ty, loc)) in &self.var_types {
      inferred_var_expr
        .entry(self.rule_loc.clone())
        .or_default()
        .entry(var.clone())
        .or_default()
        .insert(loc.clone());
      inferred_expr_types.insert(loc.clone(), ty.clone());
    }
  }

  pub fn unify_expr_types(
    &self,
    custom_types: &HashMap<String, (ValueType, Loc)>,
    constant_types: &HashMap<Loc, Type>,
    inferred_relation_types: &HashMap<String, (Vec<TypeSet>, Loc)>,
    inferred_expr_types: &mut HashMap<Loc, TypeSet>,
  ) -> Result<(), TypeInferenceError> {
    for unif in &self.unifications {
      unif.unify(
        custom_types,
        constant_types,
        inferred_relation_types,
        inferred_expr_types,
      )?;
    }
    Ok(())
  }

  pub fn propagate_variable_types(
    &self,
    inferred_var_expr: &mut HashMap<Loc, HashMap<String, BTreeSet<Loc>>>,
    inferred_expr_types: &mut HashMap<Loc, TypeSet>,
  ) -> Result<(), TypeInferenceError> {
    let mut var_tys = inferred_var_expr
      .entry(self.rule_loc.clone())
      .or_default()
      .iter()
      .map(|(var, exprs)| {
        let tys = exprs
          .iter()
          .filter_map(|e| inferred_expr_types.get(e))
          .collect::<Vec<_>>();
        if tys.is_empty() {
          Err(TypeInferenceError::UnknownVariable {
            variable: var.clone(),
            loc: self.rule_loc.clone(),
          })
        } else {
          match TypeSet::unify_type_sets(tys) {
            Ok(ty) => Ok((var.clone(), ty)),
            Err(err) => Err(err),
          }
        }
      })
      .collect::<Result<HashMap<_, _>, _>>()?;

    // Check if same variable types
    for (v1, v2) in &self.vars_of_same_type {
      let v1_ty = &var_tys[v1];
      let v2_ty = &var_tys[v2];
      match v1_ty.unify(v2_ty) {
        Err(err) => {
          let new_err = match err {
            TypeInferenceError::CannotUnifyTypes { t1, t2, .. } => TypeInferenceError::CannotUnifyVariables {
              v1: v1.clone(),
              t1,
              v2: v2.clone(),
              t2,
              loc: self.rule_loc.clone(),
            },
            err => err,
          };
          return Err(new_err);
        }
        Ok(new_ty) => {
          var_tys.insert(v1.clone(), new_ty.clone());
          var_tys.insert(v2.clone(), new_ty.clone());
        }
      }
    }

    // Check variable type constraints
    for (var, (ty, _)) in &self.var_types {
      let curr_ty = &var_tys[var];
      let new_ty = ty.unify(curr_ty)?;
      var_tys.insert(var.clone(), new_ty);
    }

    // Update the expressions that uses
    for (var, ty) in &var_tys {
      for expr_loc in &inferred_var_expr[&self.rule_loc][var] {
        inferred_expr_types.insert(expr_loc.clone(), ty.clone());
      }
    }

    // No error
    Ok(())
  }

  pub fn propagate_relation_types(
    &self,
    inferred_relation_expr: &HashMap<(String, usize), BTreeSet<Loc>>,
    inferred_expr_types: &HashMap<Loc, TypeSet>,
    inferred_relation_types: &mut HashMap<String, (Vec<TypeSet>, Loc)>,
  ) -> Result<(), TypeInferenceError> {
    // Propagate inferred relation types
    for ((predicate, i), exprs) in inferred_relation_expr {
      let tys = exprs
        .iter()
        .filter_map(|e| inferred_expr_types.get(e))
        .collect::<Vec<_>>();
      if !tys.is_empty() {
        let ty = TypeSet::unify_type_sets(tys)?;
        let arg_types = &mut inferred_relation_types.get_mut(predicate).unwrap().0;
        arg_types[(*i)] = ty;
      }
    }

    Ok(())
  }

  pub fn check_type_cast(
    &self,
    custom_types: &HashMap<String, (ValueType, Loc)>,
    inferred_expr_types: &HashMap<Loc, TypeSet>,
  ) -> Result<(), TypeInferenceError> {
    // Check if type cast can happen
    for unif in &self.unifications {
      match unif {
        Unification::TypeCast(op1, e, ty) => {
          let target_base_ty = find_value_type(custom_types, ty).unwrap();
          let op1_ty = &inferred_expr_types[op1];
          if !op1_ty.can_type_cast(&target_base_ty) {
            return Err(TypeInferenceError::CannotTypeCast {
              t1: op1_ty.clone(),
              t2: target_base_ty,
              loc: e.clone(),
            });
          }
        }
        _ => {}
      }
    }
    Ok(())
  }

  pub fn check_constraint(&self, inferred_expr_types: &HashMap<Loc, TypeSet>) -> Result<(), TypeInferenceError> {
    // Check if constraints are all boolean
    for constraint_expr in &self.constraints {
      let ty = &inferred_expr_types[constraint_expr];
      if !ty.is_boolean() {
        return Err(TypeInferenceError::ConstraintNotBoolean {
          ty: ty.clone(),
          loc: constraint_expr.clone(),
        });
      }
    }
    Ok(())
  }

  pub fn get_var_types(
    &self,
    inferred_var_expr: &HashMap<Loc, HashMap<String, BTreeSet<Loc>>>,
    inferred_expr_types: &HashMap<Loc, TypeSet>,
  ) -> HashMap<String, TypeSet> {
    inferred_var_expr[&self.rule_loc]
      .iter()
      .map(|(var, exprs)| (var.clone(), inferred_expr_types[exprs.first().unwrap()].clone()))
      .collect::<HashMap<_, _>>()
  }
}

impl NodeVisitor for LocalTypeInferenceContext {
  fn visit_atom(&mut self, atom: &Atom) {
    let pred = atom.predicate();
    self
      .atom_arities
      .entry(pred.clone())
      .or_default()
      .push((atom.arity(), atom.location().clone()));
    for (i, arg) in atom.iter_arguments().enumerate() {
      self
        .unifications
        .push(Unification::IthArgOfRelation(arg.location().clone(), pred.clone(), i));
    }
  }

  fn visit_constraint(&mut self, c: &Constraint) {
    self.constraints.push(c.expr().location().clone());
  }

  fn visit_reduce(&mut self, r: &Reduce) {
    // First check the output validity
    let vars = r.left();
    if let Some(arity) = r.operator().output_arity() {
      if vars.len() != arity {
        self.errors.push(TypeInferenceError::InvalidReduceOutput {
          op: r.operator().to_string().to_string(),
          expected: arity,
          found: r.left().len(),
          loc: r.location().clone(),
        });
        return;
      }
    }

    // Then check the number of bindings
    let maybe_num_bindings = r.operator().num_bindings();
    let bindings = r.bindings();
    if let Some(num_bindings) = maybe_num_bindings {
      if bindings.len() != num_bindings {
        self.errors.push(TypeInferenceError::InvalidReduceBindingVar {
          op: r.operator().to_string().to_string(),
          expected: num_bindings,
          found: bindings.len(),
          loc: r.location().clone(),
        });
        return;
      }
    }

    // Then propagate the variables
    match &r.operator().node {
      ReduceOperatorNode::Count => {
        if let Some(n) = vars[0].name() {
          let loc = vars[0].location();
          let ty = TypeSet::BaseType(ValueType::USize, loc.clone());
          self.var_types.insert(n.to_string(), (ty, loc.clone()));
        }
      }
      ReduceOperatorNode::Sum => {
        if let Some(n) = vars[0].name() {
          let loc = vars[0].location();
          let ty = TypeSet::Numeric(loc.clone());
          self.var_types.insert(n.to_string(), (ty, loc.clone()));

          // Result var and binding var should have the same type
          self
            .vars_of_same_type
            .push((n.to_string(), bindings[0].name().to_string()));
        }
      }
      ReduceOperatorNode::Prod => {
        if let Some(n) = vars[0].name() {
          let loc = vars[0].location();
          let ty = TypeSet::Numeric(loc.clone());
          self.var_types.insert(n.to_string(), (ty, loc.clone()));

          // Result var and binding var should have the same type
          self
            .vars_of_same_type
            .push((n.to_string(), bindings[0].name().to_string()));
        }
      }
      ReduceOperatorNode::Min => {
        if let Some(n) = vars[0].name() {
          let loc = vars[0].location();
          let ty = TypeSet::Numeric(loc.clone());
          self.var_types.insert(n.to_string(), (ty, loc.clone()));

          // Result var and binding var should have the same type
          self
            .vars_of_same_type
            .push((n.to_string(), bindings[0].name().to_string()));
        }
      }
      ReduceOperatorNode::Max => {
        if let Some(n) = vars[0].name() {
          let loc = vars[0].location();
          let ty = TypeSet::Numeric(loc.clone());
          self.var_types.insert(n.to_string(), (ty, loc.clone()));

          // Result var and binding var should have the same type
          self
            .vars_of_same_type
            .push((n.to_string(), bindings[0].name().to_string()));
        }
      }
      ReduceOperatorNode::Exists => {
        if let Some(n) = vars[0].name() {
          let loc = vars[0].location();
          let ty = TypeSet::BaseType(ValueType::Bool, loc.clone());
          self.var_types.insert(n.to_string(), (ty, loc.clone()));
        }
      }
      ReduceOperatorNode::Forall => {
        if let Some(n) = vars[0].name() {
          let loc = vars[0].location();
          let ty = TypeSet::BaseType(ValueType::Bool, loc.clone());
          self.var_types.insert(n.to_string(), (ty, loc.clone()));
        }
      }
      ReduceOperatorNode::Unique | ReduceOperatorNode::TopK(_) => {
        if vars.len() == bindings.len() {
          for (var, binding) in vars.iter().zip(bindings.iter()) {
            if let Some(n) = var.name() {
              self.vars_of_same_type.push((n.to_string(), binding.name().to_string()));
            }
          }
        } else {
          self.errors.push(TypeInferenceError::InvalidUniqueNumParams {
            num_output_vars: vars.len(),
            num_binding_vars: bindings.len(),
            loc: r.location().clone(),
          });
          return;
        }
      }
      ReduceOperatorNode::Unknown(_) => {}
    }
  }

  fn visit_variable(&mut self, v: &Variable) {
    let var_name = v.name().to_string();
    self
      .unifications
      .push(Unification::OfVariable(v.location().clone(), var_name));
  }

  fn visit_constant(&mut self, c: &Constant) {
    let type_set = TypeSet::from_constant(c);
    self
      .unifications
      .push(Unification::OfConstant(c.location().clone(), type_set));
  }

  fn visit_binary_expr(&mut self, b: &BinaryExpr) {
    let unif = if b.op().is_arith() {
      Unification::AddSubMulDivMod(
        b.op1().location().clone(),
        b.op2().location().clone(),
        b.location().clone(),
      )
    } else if b.op().is_logical() {
      Unification::AndOrXor(
        b.op1().location().clone(),
        b.op2().location().clone(),
        b.location().clone(),
      )
    } else if b.op().is_eq_neq() {
      Unification::EqNeq(
        b.op1().location().clone(),
        b.op2().location().clone(),
        b.location().clone(),
      )
    } else {
      Unification::LtLeqGtGeq(
        b.op1().location().clone(),
        b.op2().location().clone(),
        b.location().clone(),
      )
    };
    self.unifications.push(unif);
  }

  fn visit_unary_expr(&mut self, u: &UnaryExpr) {
    let unif = if u.op().is_pos_neg() {
      Unification::PosNeg(u.op1().location().clone(), u.location().clone())
    } else if u.op().is_not() {
      Unification::Not(u.op1().location().clone(), u.location().clone())
    } else {
      let ty = u.op().cast_to_type().unwrap();
      Unification::TypeCast(u.op1().location().clone(), u.location().clone(), ty.clone())
    };
    self.unifications.push(unif)
  }

  fn visit_if_then_else_expr(&mut self, i: &IfThenElseExpr) {
    let unif = Unification::IfThenElse(
      i.location().clone(),
      i.cond().location().clone(),
      i.then_br().location().clone(),
      i.else_br().location().clone(),
    );
    self.unifications.push(unif)
  }

  fn visit_call_expr(&mut self, c: &CallExpr) {
    match c.function().function() {
      Some(f) => {
        let unif = Unification::Call(
          f,
          c.iter_args().map(|a| a.location().clone()).collect(),
          c.location().clone(),
        );
        self.unifications.push(unif)
      }
      _ => {}
    }
  }
}
