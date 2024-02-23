use std::collections::*;

use super::*;
use crate::common::binary_op::BinaryOp;
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
  pub errors: Vec<Error>,
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
    r.walk(&mut ctx);
    ctx
  }

  pub fn from_reduce_rule(r: &ReduceRule) -> Self {
    let mut ctx = Self::new(r.location().clone());
    r.walk(&mut ctx);
    ctx
  }

  pub fn from_atom(a: &Atom) -> Self {
    let mut ctx = Self::new(a.location().clone());
    a.walk(&mut ctx);
    ctx
  }

  pub fn unify_atom_arities(
    &self,
    predicate_registry: &PredicateTypeRegistry,
    inferred_relation_types: &mut HashMap<String, (Vec<TypeSet>, Loc)>,
  ) -> Result<(), Error> {
    for (pred, arities) in &self.atom_arities {
      // Skip foreign predicates
      if predicate_registry.contains_predicate(pred) {
        continue;
      }

      // Make sure we have inferred relation types for the predicate
      if !inferred_relation_types.contains_key(pred) {
        let (arity, atom_loc) = &arities[0];
        let init_types = vec![TypeSet::Any(Loc::default()); *arity];
        inferred_relation_types.insert(pred.clone(), (init_types, atom_loc.clone()));
      }

      // Make sure the arity matches
      let (tys, _) = &inferred_relation_types[pred];
      for (arity, atom_loc) in arities {
        if arity != &tys.len() {
          return Err(Error::arity_mismatch(pred.clone(), tys.len(), *arity, atom_loc.clone()));
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
    function_type_registry: &FunctionTypeRegistry,
    predicate_type_registry: &PredicateTypeRegistry,
    aggregate_type_registry: &AggregateTypeRegistry,
    inferred_expr_types: &mut HashMap<Loc, TypeSet>,
  ) -> Result<(), Error> {
    for unif in &self.unifications {
      unif.unify(
        custom_types,
        constant_types,
        inferred_relation_types,
        function_type_registry,
        predicate_type_registry,
        aggregate_type_registry,
        inferred_expr_types,
      )?;
    }
    Ok(())
  }

  pub fn propagate_variable_types(
    &self,
    inferred_var_expr: &mut HashMap<Loc, HashMap<String, BTreeSet<Loc>>>,
    inferred_expr_types: &mut HashMap<Loc, TypeSet>,
  ) -> Result<(), Error> {
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
          Err(Error::unknown_variable(var.clone(), self.rule_loc.clone()))
        } else {
          match TypeSet::unify_type_sets(tys) {
            Ok(ty) => Ok((var.clone(), ty)),
            Err(err) => Err(err.into()),
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
          return Err(Error::cannot_unify_variables(
            v1.clone(),
            err.t1,
            v2.clone(),
            err.t2,
            self.rule_loc.clone(),
          ));
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
      let new_ty = ty.unify(curr_ty).map_err(|e| e.into())?;
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
  ) -> Result<(), Error> {
    // Propagate inferred relation types
    for ((predicate, i), exprs) in inferred_relation_expr {
      let tys = exprs
        .iter()
        .filter_map(|e| inferred_expr_types.get(e))
        .collect::<Vec<_>>();
      if !tys.is_empty() {
        let ty = TypeSet::unify_type_sets(tys).map_err(|e| e.into())?;
        if let Some((arg_types, _)) = inferred_relation_types.get_mut(predicate) {
          arg_types[*i] = ty;
        }
      }
    }

    Ok(())
  }

  pub fn check_type_cast(
    &self,
    custom_types: &HashMap<String, (ValueType, Loc)>,
    inferred_expr_types: &HashMap<Loc, TypeSet>,
  ) -> Result<(), Error> {
    // Check if type cast can happen
    for unif in &self.unifications {
      match unif {
        Unification::TypeCast(op1, e, ty) => {
          let target_base_ty = find_value_type(custom_types, ty).unwrap();
          let op1_ty = &inferred_expr_types[op1];
          if !op1_ty.can_type_cast(&target_base_ty) {
            return Err(Error::cannot_type_cast(op1_ty.clone(), target_base_ty, e.clone()));
          }
        }
        _ => {}
      }
    }
    Ok(())
  }

  pub fn check_constraint(&self, inferred_expr_types: &HashMap<Loc, TypeSet>) -> Result<(), Error> {
    // Check if constraints are all boolean
    for constraint_expr in &self.constraints {
      let ty = &inferred_expr_types[constraint_expr];
      if !ty.is_boolean() {
        return Err(Error::constraint_not_boolean(ty.clone(), constraint_expr.clone()));
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

impl NodeVisitor<Atom> for LocalTypeInferenceContext {
  fn visit(&mut self, atom: &Atom) {
    let pred = atom.formatted_predicate();
    self
      .atom_arities
      .entry(pred.clone())
      .or_default()
      .push((atom.arity(), atom.location().clone()));
    for (i, arg) in atom.iter_args().enumerate() {
      self
        .unifications
        .push(Unification::IthArgOfRelation(arg.location().clone(), pred.clone(), i));
    }
  }
}

impl NodeVisitor<Constraint> for LocalTypeInferenceContext {
  fn visit(&mut self, c: &Constraint) {
    self.constraints.push(c.expr().location().clone());
  }
}

impl NodeVisitor<Reduce> for LocalTypeInferenceContext {
  fn visit(&mut self, r: &Reduce) {
    // First get the aggregate type
    let agg_op = r.operator();
    let agg_name = agg_op.name().name();
    let has_exclamation_mark = agg_op.has_exclaimation_mark().clone();

    // Add the aggregation unification
    self.unifications.push(Unification::Aggregate {
      left_vars: r.iter_left().map(|vow| vow.location()).cloned().collect(),
      aggregate_name: agg_name.clone(),
      aggregate: agg_op.location().clone(),
      params: agg_op
        .parameters()
        .iter()
        .filter_map(|param| match param {
          ReduceParam::Positional(c) => Some(c.location()),
          _ => None,
        })
        .cloned()
        .collect(),
      named_params: agg_op
        .parameters()
        .iter()
        .filter_map(|param| match param {
          ReduceParam::Named(c) => {
            let name = c.name().name().clone();
            let name_loc = c.name().location().clone();
            let value_loc = c.value().location().clone();
            Some((name, (name_loc, value_loc)))
          }
          _ => None,
        })
        .collect(),
      arg_vars: r.iter_args().map(|a| a.location()).cloned().collect(),
      input_vars: r.iter_bindings().map(|a| a.location()).cloned().collect(),
      has_exclamation_mark,
    });
  }
}

impl NodeVisitor<Variable> for LocalTypeInferenceContext {
  fn visit(&mut self, v: &Variable) {
    let var_name = v.name().to_string();
    self
      .unifications
      .push(Unification::OfVariable(v.location().clone(), var_name));
  }
}

impl NodeVisitor<Constant> for LocalTypeInferenceContext {
  fn visit(&mut self, c: &Constant) {
    let type_set = TypeSet::from_constant(c);
    self
      .unifications
      .push(Unification::OfConstant(c.location().clone(), type_set));
  }
}

impl NodeVisitor<BinaryExpr> for LocalTypeInferenceContext {
  fn visit(&mut self, b: &BinaryExpr) {
    let op1 = b.op1().location().clone();
    let op2 = b.op2().location().clone();
    let loc = b.location().clone();
    let unif = match b.op().op() {
      BinaryOp::Add => Unification::Add(op1, op2, loc),
      BinaryOp::Sub => Unification::Sub(op1, op2, loc),
      BinaryOp::Mul => Unification::Mult(op1, op2, loc),
      BinaryOp::Div => Unification::Div(op1, op2, loc),
      BinaryOp::Mod => Unification::Mod(op1, op2, loc),
      BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => Unification::AndOrXor(op1, op2, loc),
      BinaryOp::Eq | BinaryOp::Neq => Unification::EqNeq(op1, op2, loc),
      BinaryOp::Lt | BinaryOp::Leq | BinaryOp::Gt | BinaryOp::Geq => Unification::LtLeqGtGeq(op1, op2, loc),
    };
    self.unifications.push(unif);
  }
}

impl NodeVisitor<UnaryExpr> for LocalTypeInferenceContext {
  fn visit(&mut self, u: &UnaryExpr) {
    let unif = if u.op().is_pos_neg() {
      Unification::PosNeg(u.op1().location().clone(), u.location().clone())
    } else if u.op().is_not() {
      Unification::Not(u.op1().location().clone(), u.location().clone())
    } else {
      let ty = u.op().as_typecast().unwrap();
      Unification::TypeCast(u.op1().location().clone(), u.location().clone(), ty.clone())
    };
    self.unifications.push(unif)
  }
}

impl NodeVisitor<IfThenElseExpr> for LocalTypeInferenceContext {
  fn visit(&mut self, i: &IfThenElseExpr) {
    let unif = Unification::IfThenElse(
      i.location().clone(),
      i.cond().location().clone(),
      i.then_br().location().clone(),
      i.else_br().location().clone(),
    );
    self.unifications.push(unif)
  }
}

impl NodeVisitor<CallExpr> for LocalTypeInferenceContext {
  fn visit(&mut self, c: &CallExpr) {
    let unif = Unification::Call(
      c.function_identifier().name().to_string(),
      c.iter_args().map(|a| a.location().clone()).collect(),
      c.location().clone(),
    );
    self.unifications.push(unif)
  }
}

impl NodeVisitor<NewExpr> for LocalTypeInferenceContext {
  fn visit(&mut self, n: &NewExpr) {
    let unif = Unification::New(
      n.functor_name().to_string(),
      n.iter_args().map(|a| a.location().clone()).collect(),
      n.location().clone(),
    );
    self.unifications.push(unif)
  }
}
