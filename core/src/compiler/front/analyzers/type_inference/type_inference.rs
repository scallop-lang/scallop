use std::collections::*;

use crate::common::foreign_function::ForeignFunctionRegistry;
use crate::common::tuple_type::*;
use crate::common::value_type::*;
use crate::compiler::front::*;

use super::*;

#[derive(Clone, Debug)]
pub struct TypeInference {
  pub custom_types: HashMap<String, (ValueType, Loc)>,
  pub constant_types: HashMap<Loc, Type>,
  pub function_type_registry: FunctionTypeRegistry,
  pub relation_type_decl_loc: HashMap<String, Loc>,
  pub inferred_relation_types: HashMap<String, (Vec<TypeSet>, Loc)>,
  pub rule_variable_type: HashMap<Loc, HashMap<String, TypeSet>>,
  pub rule_local_contexts: Vec<LocalTypeInferenceContext>,
  pub query_relations: HashMap<String, Loc>,
  pub expr_types: HashMap<Loc, TypeSet>,
  pub errors: Vec<TypeInferenceError>,
}

impl TypeInference {
  pub fn new(function_registry: &ForeignFunctionRegistry) -> Self {
    Self {
      custom_types: HashMap::new(),
      constant_types: HashMap::new(),
      function_type_registry: FunctionTypeRegistry::from_foreign_function_registry(function_registry),
      relation_type_decl_loc: HashMap::new(),
      inferred_relation_types: HashMap::new(),
      rule_variable_type: HashMap::new(),
      rule_local_contexts: Vec::new(),
      query_relations: HashMap::new(),
      expr_types: HashMap::new(),
      errors: vec![],
    }
  }

  pub fn expr_value_type<T>(&self, t: &T) -> Option<ValueType>
  where
    T: WithLocation,
  {
    self.expr_types.get(t.location()).map(TypeSet::to_default_value_type)
  }

  pub fn num_relations(&self) -> usize {
    self
      .inferred_relation_types
      .iter()
      .filter(|(n, _)| !n.contains("#"))
      .count()
  }

  pub fn extend_constant_types(&mut self, constant_types: HashMap<Loc, Type>) {
    self.constant_types.extend(constant_types.into_iter());
  }

  pub fn relations(&self) -> Vec<String> {
    self
      .inferred_relation_types
      .iter()
      .filter_map(|(n, _)| if !n.contains("#") { Some(n.clone()) } else { None })
      .collect()
  }

  pub fn has_relation(&self, relation: &str) -> bool {
    self.inferred_relation_types.contains_key(relation)
  }

  pub fn relation_arg_types(&self, relation: &str) -> Option<Vec<ValueType>> {
    let inferred_relation_types = &self.inferred_relation_types;
    if let Some((tys, _)) = &inferred_relation_types.get(relation) {
      Some(tys.iter().map(type_inference::TypeSet::to_default_value_type).collect())
    } else {
      None
    }
  }

  pub fn relation_tuple_type(&self, relation: &str) -> Option<TupleType> {
    self
      .relation_arg_types(relation)
      .map(|a| TupleType::from_types(&a, false))
  }

  pub fn variable_type(&self, rule_loc: &Loc, var: &str) -> ValueType {
    self.rule_variable_type[rule_loc][var].to_default_value_type()
  }

  pub fn variable_types<'a, I, T>(&self, rule_loc: &Loc, vars: I) -> Vec<ValueType>
  where
    I: Iterator<Item = T>,
    T: Into<&'a String>,
  {
    vars
      .map(|v| self.rule_variable_type[rule_loc][v.into()].to_default_value_type())
      .collect()
  }

  pub fn find_value_type(&self, ty: &Type) -> Result<ValueType, TypeInferenceError> {
    find_value_type(&self.custom_types, ty)
  }

  pub fn check_and_add_custom_type(&mut self, name: &str, ty: &Type, loc: &Loc) {
    if self.custom_types.contains_key(name) {
      let (_, source_loc) = &self.custom_types[name];
      self.errors.push(TypeInferenceError::DuplicateTypeDecl {
        type_name: name.to_string(),
        source_decl_loc: source_loc.clone(),
        duplicate_decl_loc: loc.clone(),
      });
    } else {
      match self.find_value_type(ty) {
        Ok(base_ty) => {
          self.custom_types.insert(name.to_string(), (base_ty, loc.clone()));
        }
        Err(err) => {
          self.errors.push(err);
        }
      }
    }
  }

  pub fn check_and_add_relation_type<'a>(&mut self, predicate: &str, tys: impl Iterator<Item = &'a Type>, loc: &Loc) {
    // Check if the relation has been declared
    if self.relation_type_decl_loc.contains_key(predicate) {
      let source_loc = &self.relation_type_decl_loc[predicate];
      self.errors.push(TypeInferenceError::DuplicateRelationTypeDecl {
        predicate: predicate.to_string(),
        source_decl_loc: source_loc.clone(),
        duplicate_decl_loc: loc.clone(),
      });
      return;
    }

    // Add the declaration
    self.relation_type_decl_loc.insert(predicate.to_string(), loc.clone());

    // Add the declaration to the inferred types
    let tys = tys.collect::<Vec<_>>();
    let maybe_tys = tys
      .iter()
      .map(|ty| match self.find_value_type(ty) {
        Ok(t) => Ok(TypeSet::BaseType(t, ty.location().clone())),
        Err(err) => Err(err),
      })
      .collect::<Result<Vec<_>, _>>();
    match maybe_tys {
      Ok(tys) => {
        // Check if there is existing inferred types about this relation
        let new_type_sets = if self.inferred_relation_types.contains_key(predicate) {
          let (existing, _) = &self.inferred_relation_types[predicate];
          let maybe_new_type_sets = existing
            .iter()
            .zip(tys.iter())
            .map(|(t1, t2)| t1.unify(t2))
            .collect::<Result<Vec<TypeSet>, _>>();
          match maybe_new_type_sets {
            Ok(new_type_sets) => new_type_sets,
            Err(err) => {
              self.errors.push(err);
              return;
            }
          }
        } else {
          tys
        };

        // Alwasy overwrite the previous since this is a declaration
        self
          .inferred_relation_types
          .insert(predicate.to_string(), (new_type_sets, loc.clone()));
      }
      Err(err) => {
        self.errors.push(err);
      }
    }
  }

  pub fn resolve_constant_type(&self, c: &Constant) -> Result<TypeSet, TypeInferenceError> {
    if let Some(ty) = self.constant_types.get(c.location()) {
      let val_ty = find_value_type(&self.custom_types, ty)?;
      Ok(TypeSet::BaseType(val_ty, ty.location().clone()))
    } else {
      Ok(TypeSet::from_constant(c))
    }
  }

  pub fn check_query_predicates(&mut self) {
    for (pred, loc) in &self.query_relations {
      if !self.inferred_relation_types.contains_key(pred) {
        self.errors.push(TypeInferenceError::UnknownQueryRelationType {
          predicate: pred.clone(),
          loc: loc.clone(),
        });
      }
    }
  }

  pub fn infer_types(&mut self) {
    if let Err(err) = self.infer_types_helper() {
      self.errors.push(err);
    }
  }

  fn infer_types_helper(&mut self) -> Result<(), TypeInferenceError> {
    // Mapping from variable to set of expressions
    // Mapping from relation argument to set of expressions
    let mut inferred_var_expr = HashMap::<Loc, HashMap<String, BTreeSet<Loc>>>::new();
    let mut inferred_relation_expr = HashMap::<(String, usize), BTreeSet<Loc>>::new();

    // Fixpoint states
    let mut old_inferred_expr_types = HashMap::<Loc, TypeSet>::new();
    let mut inferred_expr_types = HashMap::<Loc, TypeSet>::new();

    // Initial facts
    for ctx in &self.rule_local_contexts {
      ctx.populate_inference_data(
        &mut inferred_relation_expr,
        &mut inferred_var_expr,
        &mut inferred_expr_types,
      );
    }

    // Fixpoint iteration
    let mut first_time = true;
    while first_time || old_inferred_expr_types != inferred_expr_types {
      first_time = false;
      old_inferred_expr_types = inferred_expr_types.clone();

      // Go through all unifications
      for ctx in &self.rule_local_contexts {
        ctx.unify_expr_types(
          &self.custom_types,
          &self.constant_types,
          &self.inferred_relation_types,
          &self.function_type_registry,
          &mut inferred_expr_types,
        )?;
        ctx.propagate_variable_types(&mut inferred_var_expr, &mut inferred_expr_types)?;
        ctx.propagate_relation_types(
          &inferred_relation_expr,
          &inferred_expr_types,
          &mut self.inferred_relation_types,
        )?;
      }
    }

    // Final step, iterate through each rule and their local context
    // and make sure everything is fine
    for ctx in &self.rule_local_contexts {
      ctx.check_type_cast(&self.custom_types, &inferred_expr_types)?;
      ctx.check_constraint(&inferred_expr_types)?;

      // Get variable type mapping and store it
      let var_ty = ctx.get_var_types(&inferred_var_expr, &inferred_expr_types);
      self.rule_variable_type.insert(ctx.rule_loc.clone(), var_ty);
    }

    // Add expression types
    self.expr_types = inferred_expr_types;

    Ok(())
  }
}

impl NodeVisitor for TypeInference {
  fn visit_subtype_decl(&mut self, subtype_decl: &SubtypeDecl) {
    self.check_and_add_custom_type(subtype_decl.name(), subtype_decl.subtype_of(), subtype_decl.location());
  }

  fn visit_alias_type_decl(&mut self, alias_type_decl: &AliasTypeDecl) {
    self.check_and_add_custom_type(
      alias_type_decl.name(),
      alias_type_decl.alias_of(),
      alias_type_decl.location(),
    );
  }

  fn visit_relation_type(&mut self, relation_type: &RelationType) {
    self.check_and_add_relation_type(
      relation_type.predicate(),
      relation_type.arg_types(),
      relation_type.location(),
    );
  }

  fn visit_const_assignment(&mut self, const_assign: &ConstAssignment) {
    if let Some(raw_type) = const_assign.ty() {
      let result = find_value_type(&self.custom_types, raw_type).and_then(|ty| {
        let ts = TypeSet::from_constant(const_assign.value());
        ts.unify(&TypeSet::BaseType(ty, raw_type.location().clone()))
      });
      match result {
        Ok(_) => {}
        Err(mut err) => {
          err.annotate_location(const_assign.location());
          self.errors.push(err);
        }
      }
    }
  }

  fn visit_constant_set_decl(&mut self, constant_set_decl: &ConstantSetDecl) {
    let pred = constant_set_decl.predicate();

    // There's nothing we can check if there is no tuple inside the set
    if constant_set_decl.num_tuples() == 0 {
      return;
    }

    // First get the arity of the constant set.
    let arity = {
      // Compute the arity from the set
      let maybe_arity = constant_set_decl.iter_tuples().fold(Ok(None), |acc, tuple| match acc {
        Ok(maybe_arity) => {
          let current_arity = tuple.arity();
          if let Some(previous_arity) = &maybe_arity {
            if previous_arity != &current_arity {
              return Err(TypeInferenceError::ConstantSetArityMismatch {
                predicate: pred.clone(),
                decl_loc: constant_set_decl.location().clone(),
                mismatch_tuple_loc: tuple.location().clone(),
              });
            } else {
              Ok(Some(current_arity))
            }
          } else {
            Ok(Some(current_arity))
          }
        }
        Err(err) => Err(err),
      });

      // If there is arity mismatch inside the set, add the error and stop
      match maybe_arity {
        Err(err) => {
          self.errors.push(err);
          return;
        }
        Ok(arity) => arity.unwrap(),
      }
    };

    // Get the type set to start with
    let (mut type_sets, loc) = if self.inferred_relation_types.contains_key(pred) {
      self.inferred_relation_types[pred].clone()
    } else {
      let type_sets = vec![TypeSet::Any(Loc::default()); arity];
      let loc = constant_set_decl.location().clone();
      (type_sets, loc)
    };

    // Then iterate through the tuples to unify the constant types
    for tuple in constant_set_decl.iter_tuples() {
      // Check if the arity of the tuple matches the defined ones
      if tuple.arity() != type_sets.len() {
        self.errors.push(TypeInferenceError::ArityMismatch {
          predicate: pred.clone(),
          expected: type_sets.len(),
          actual: tuple.arity(),
          source_loc: loc.clone(),
          mismatch_loc: tuple.location().clone(),
        });
        continue;
      }

      // If matches, we check whether the type matches
      for (c, ts) in tuple.iter_constants().zip(type_sets.iter_mut()) {
        // Unwrap is okay here because we have checked for constant in the pre-transformation analysis
        let curr_ts = match self.resolve_constant_type(c.constant().unwrap()) {
          Ok(t) => t,
          Err(err) => {
            self.errors.push(err);
            continue;
          }
        };
        let maybe_new_ts = ts.unify(&curr_ts);
        match maybe_new_ts {
          Ok(new_ts) => {
            *ts = new_ts;
          }
          Err(err) => {
            self.errors.push(err);
            return;
          }
        }
      }
    }

    // Finally update the type set
    if self.inferred_relation_types.contains_key(pred) {
      self.inferred_relation_types.get_mut(pred).unwrap().0 = type_sets;
    } else {
      self.inferred_relation_types.insert(pred.clone(), (type_sets, loc));
    }
  }

  fn visit_fact_decl(&mut self, fact_decl: &FactDecl) {
    let pred = fact_decl.predicate();
    let maybe_curr_type_sets = fact_decl
      .iter_arguments()
      .map(|arg| match arg {
        Expr::Constant(c) => self.resolve_constant_type(c),
        _ => {
          panic!("[Internal Error] Non-constant occurring in fact decl during type inference. This should not happen")
        }
      })
      .collect::<Result<Vec<_>, _>>();
    let curr_type_sets = match maybe_curr_type_sets {
      Ok(t) => t,
      Err(err) => {
        self.errors.push(err);
        return;
      }
    };

    // Check the type
    if self.inferred_relation_types.contains_key(pred) {
      let (original_type_sets, original_type_def_loc) = &self.inferred_relation_types[pred];

      // First check if the arity matches
      if curr_type_sets.len() != original_type_sets.len() {
        self.errors.push(TypeInferenceError::ArityMismatch {
          predicate: pred.clone(),
          expected: original_type_sets.len(),
          actual: curr_type_sets.len(),
          source_loc: original_type_def_loc.clone(),
          mismatch_loc: fact_decl.atom().location().clone(),
        });
        return;
      }

      // If matches, unify the types
      let maybe_new_type_sets = original_type_sets
        .iter()
        .zip(curr_type_sets.iter())
        .map(|(orig_ts, new_ts)| orig_ts.unify(new_ts))
        .collect::<Result<Vec<_>, _>>();
      match maybe_new_type_sets {
        Ok(new_type_sets) => {
          self.inferred_relation_types.get_mut(pred).unwrap().0 = new_type_sets;
        }
        Err(err) => self.errors.push(err),
      }
    } else {
      self
        .inferred_relation_types
        .insert(pred.clone(), (curr_type_sets, fact_decl.location().clone()));
    }
  }

  fn visit_rule(&mut self, rule: &Rule) {
    let ctx = LocalTypeInferenceContext::from_rule(rule);

    // Check if context has error already
    if !ctx.errors.is_empty() {
      self.errors.extend(ctx.errors);
      return;
    }

    // First unify atom arity
    if let Err(err) = ctx.unify_atom_arities(&mut self.inferred_relation_types) {
      self.errors.push(err);
      return;
    }

    // Add the context
    self.rule_local_contexts.push(ctx);
  }

  fn visit_query(&mut self, query: &Query) {
    match &query.node {
      QueryNode::Atom(atom) => {
        let ctx = LocalTypeInferenceContext::from_atom(atom);

        // Check if context has error already
        if !ctx.errors.is_empty() {
          self.errors.extend(ctx.errors);
          return;
        }

        // Unify atom arity
        if let Err(err) = ctx.unify_atom_arities(&mut self.inferred_relation_types) {
          self.errors.push(err);
          return;
        }

        // Add the context
        self.rule_local_contexts.push(ctx);
      }
      QueryNode::Predicate(p) => {
        self.query_relations.insert(p.name().to_string(), p.location().clone());
      }
    }
  }
}

pub fn find_value_type(
  custom_types: &HashMap<String, (ValueType, Loc)>,
  ty: &Type,
) -> Result<ValueType, TypeInferenceError> {
  match ty.to_value_type() {
    Ok(base_ty) => Ok(base_ty),
    Err(other_name) => {
      if custom_types.contains_key(&other_name) {
        let base_ty = custom_types[&other_name].0.clone();
        Ok(base_ty)
      } else {
        Err(TypeInferenceError::UnknownCustomType {
          type_name: other_name,
          loc: ty.location().clone(),
        })
      }
    }
  }
}
