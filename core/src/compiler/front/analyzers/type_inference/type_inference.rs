use std::collections::*;

use crate::common::adt_variant_registry::ADTVariantRegistry;
use crate::common::foreign_function::*;
use crate::common::foreign_predicate::*;
use crate::common::tuple_type::*;
use crate::common::value_type::*;
use crate::compiler::front::*;

use super::*;

#[derive(Clone, Debug)]
pub struct TypeInference {
  /// A mapping from the custom type name to its inferred value type and decl location
  pub custom_types: HashMap<String, (ValueType, Loc)>,

  /// A mapping from referred constant variables' location to its declared type, if the type is specified in the constant declaration
  pub constant_types: HashMap<Loc, Type>,

  /// Foreign function types
  pub foreign_function_type_registry: FunctionTypeRegistry,

  /// Foreign predicate types
  pub foreign_predicate_type_registry: PredicateTypeRegistry,

  /// A mapping from relation name to its type declaration location
  pub relation_type_decl_loc: HashMap<String, Loc>,

  /// Relation field names
  pub relation_field_names: HashMap<String, Vec<Option<String>>>,

  /// A mapping from internal relation name to ADT variant name, e.g. `adt#Node` -> `Node`
  pub adt_relations: HashMap<String, (String, Vec<bool>)>,

  /// A mapping from relation name to its argument types `Vec<TypeSet>` and the location `Loc` where such type is inferred
  pub inferred_relation_types: HashMap<String, (Vec<TypeSet>, Loc)>,

  /// A mapping { Rule Location: { Variable Name: Inferred Type } }
  pub rule_variable_type: HashMap<Loc, HashMap<String, TypeSet>>,

  /// The local inference contexts of a rule
  pub rule_local_contexts: Vec<LocalTypeInferenceContext>,

  /// A mapping from a relation name that is queried to the location where it is queried
  pub query_relations: HashMap<String, Loc>,

  /// A mapping from expression ID to its inferred types
  pub expr_types: HashMap<Loc, TypeSet>,

  /// A list of errors obtained from the type inference process
  pub errors: Vec<TypeInferenceError>,
}

impl TypeInference {
  pub fn new(function_registry: &ForeignFunctionRegistry, predicate_registry: &ForeignPredicateRegistry) -> Self {
    Self {
      custom_types: HashMap::new(),
      constant_types: HashMap::new(),
      foreign_function_type_registry: FunctionTypeRegistry::from_foreign_function_registry(function_registry),
      foreign_predicate_type_registry: PredicateTypeRegistry::from_foreign_predicate_registry(predicate_registry),
      relation_type_decl_loc: HashMap::new(),
      relation_field_names: HashMap::new(),
      adt_relations: HashMap::new(),
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
    T: AstNode,
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
    if self.query_relations.is_empty() {
      self
        .inferred_relation_types
        .iter()
        .filter_map(|(n, _)| if !n.contains("#") { Some(n.clone()) } else { None })
        .collect()
    } else {
      self.query_relations.iter().map(|(n, _)| n.clone()).collect()
    }
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

  pub fn relation_field_names(&self, relation: &str) -> Option<&Vec<Option<String>>> {
    self.relation_field_names.get(relation)
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

  pub fn check_and_add_relation_type<'a>(&mut self, predicate: &str, tys: &Vec<ArgTypeBinding>, loc: &Loc) {
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

    // Add the relation field names
    if tys.iter().any(|a| a.has_name()) {
      let field_names = tys.iter().map(|a| a.name().as_ref().map(|n| n.to_string())).collect();
      self.relation_field_names.insert(predicate.to_string(), field_names);
    }

    // Add the declaration to the inferred types
    let maybe_tys = tys
      .iter()
      .map(|arg| {
        let ty = arg.ty();
        match self.find_value_type(ty) {
          Ok(t) => Ok(TypeSet::BaseType(t, ty.location().clone())),
          Err(err) => Err(err),
        }
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
    let mut inferred_var_expr = HashMap::<Loc, HashMap<String, BTreeSet<Loc>>>::new();

    // Mapping from relation argument to set of expressions
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
          &self.foreign_function_type_registry,
          &self.foreign_predicate_type_registry,
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

  pub fn create_adt_variant_registry(&self) -> ADTVariantRegistry {
    // Create an empty registry
    let mut registry = ADTVariantRegistry::new();

    // Iterate through all the ADT relations and add them to the registry
    for (relation_name, (variant_name, _)) in &self.adt_relations {
      let arg_types = self
        .relation_arg_types(relation_name)
        .expect("[Internal Error] expect adt variant type to be inferred");
      registry.add(variant_name.to_string(), relation_name.to_string(), arg_types);
    }

    // Return the registry
    registry
  }
}

impl NodeVisitor<SubtypeDecl> for TypeInference {
  fn visit(&mut self, subtype_decl: &SubtypeDecl) {
    self.check_and_add_custom_type(subtype_decl.name().name(), subtype_decl.subtype_of(), subtype_decl.location());
  }
}

impl NodeVisitor<AliasTypeDecl> for TypeInference {
  fn visit(&mut self, alias_type_decl: &AliasTypeDecl) {
    self.check_and_add_custom_type(
      alias_type_decl.name().name(),
      alias_type_decl.alias_of(),
      alias_type_decl.location(),
    );
  }
}

impl NodeVisitor<RelationTypeDecl> for TypeInference {
  fn visit(&mut self, relation_type_decl: &RelationTypeDecl) {
    if let Some(attr) = relation_type_decl.attrs().find("adt") {
      // Get the variant name string
      let adt_variant_name = attr
        .pos_arg_to_string(0)
        .expect("[Internal Error] internally annotated adt attribute does not have a string as the argument 0");

      // Get the adt variant
      let adt_variant_relation_name = format!("adt#{adt_variant_name}");

      // Get the is_entity list
      let adt_is_entity_list: Vec<bool> = attr
        .pos_arg_to_list(1)
        .expect("[Internal Error] internally annotated adt attribute does not have a list of boolean as the argument 1")
        .iter()
        .map(|arg| {
          arg
            .as_boolean()
            .expect(
              "[Internal Error] internally annotated adt attribute does not have a list of boolean as the argument 1",
            )
            .clone()
        })
        .collect();

      // Add the adt annotation into the `adt_relations` mapping
      self.adt_relations.insert(
        adt_variant_relation_name,
        (adt_variant_name.clone(), adt_is_entity_list),
      );
    }
  }
}

impl NodeVisitor<RelationType> for TypeInference {
  fn visit(&mut self, relation_type: &RelationType) {
    // Check if the relation is a foreign predicate
    let predicate = relation_type.predicate_name();
    if self.foreign_predicate_type_registry.contains_predicate(predicate) {
      self.errors.push(TypeInferenceError::CannotRedefineForeignPredicate {
        pred: predicate.to_string(),
        loc: relation_type.location().clone(),
      });
      return;
    }

    self.check_and_add_relation_type(
      relation_type.predicate_name(),
      relation_type.arg_bindings(),
      relation_type.location(),
    );
  }
}

impl NodeVisitor<EnumTypeDecl> for TypeInference {
  fn visit(&mut self, enum_type_decl: &EnumTypeDecl) {
    // First add the enum type
    let ty = Type::usize();
    self.check_and_add_custom_type(enum_type_decl.name().name(), &ty, enum_type_decl.location());

    // And then declare all the constant types
    // Note: we do not check for duplicated names here, as they are handled by `ConstantDeclAnalysis`.
    for member in enum_type_decl.iter_members() {
      match member.assigned_num() {
        Some(c) => match c {
          Constant::Integer(i) => {
            if i.int() < &0 {
              self.errors.push(TypeInferenceError::NegativeEnumValue {
                found: i.int().clone(),
                loc: c.location().clone(),
              })
            }
          }
          _ => self.errors.push(TypeInferenceError::BadEnumValueKind {
            found: c.kind(),
            loc: c.location().clone(),
          }),
        },
        _ => {}
      }
    }
  }
}

impl NodeVisitor<FunctionTypeDecl> for TypeInference {
  fn visit(&mut self, _: &FunctionTypeDecl) {
    // TODO
    println!("[Warning] Cannot handle function type declaration yet; the declarations should be processed by external attributes")
  }
}

impl NodeVisitor<ConstAssignment> for TypeInference {
  fn visit(&mut self, const_assign: &ConstAssignment) {
    if let Some(raw_type) = const_assign.ty() {
      let result = find_value_type(&self.custom_types, raw_type).and_then(|ty| {
        let ts = TypeSet::from_constant(const_assign.value().as_constant().expect("[Internal Error] During type inference, all entities should be normalized to constant. This is probably an internal error."));
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
}

impl NodeVisitor<ConstantSetDecl> for TypeInference {
  fn visit(&mut self, constant_set_decl: &ConstantSetDecl) {
    let pred = constant_set_decl.predicate_name();

    // Check if the relation is a foreign predicate
    if self.foreign_predicate_type_registry.contains_predicate(pred) {
      self.errors.push(TypeInferenceError::CannotRedefineForeignPredicate {
        pred: pred.to_string(),
        loc: constant_set_decl.location().clone(),
      });
      return;
    }

    // There's nothing we can check if there is no tuple inside the set
    if constant_set_decl.set().num_tuples() == 0 {
      return;
    }

    // First get the arity of the constant set.
    let arity = {
      // Compute the arity from the set
      let maybe_arity = constant_set_decl.set().iter_tuples().fold(Ok(None), |acc, tuple| match acc {
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
    for tuple in constant_set_decl.set().iter_tuples() {
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
        let curr_ts = match self.resolve_constant_type(c.as_constant().unwrap()) {
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
}

impl NodeVisitor<FactDecl> for TypeInference {
  fn visit(&mut self, fact_decl: &FactDecl) {
    let pred = fact_decl.predicate_name();

    // Check if the relation is a foreign predicate
    if self.foreign_predicate_type_registry.contains_predicate(&pred) {
      self.errors.push(TypeInferenceError::CannotRedefineForeignPredicate {
        pred: pred.to_string(),
        loc: fact_decl.location().clone(),
      });
      return;
    }

    // Check if the relation is an ADT
    if pred.contains("adt#") {
      // Make sure that the predicate is an existing ADT relation
      if !self.adt_relations.contains_key(pred) {
        self.errors.push(TypeInferenceError::UnknownADTVariant {
          predicate: pred[4..].to_string(),
          loc: fact_decl.atom().predicate().location().clone(),
        })
      }
    }

    let maybe_curr_type_sets = fact_decl
      .iter_args()
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
        if let Some((variant_name, _)) = self.adt_relations.get(pred) {
          self.errors.push(TypeInferenceError::ADTVariantArityMismatch {
            variant: variant_name.clone(),
            expected: original_type_sets.len() - 1,
            actual: curr_type_sets.len() - 1,
            loc: fact_decl.atom().location().clone(),
          });
        } else {
          self.errors.push(TypeInferenceError::ArityMismatch {
            predicate: pred.clone(),
            expected: original_type_sets.len(),
            actual: curr_type_sets.len(),
            source_loc: original_type_def_loc.clone(),
            mismatch_loc: fact_decl.atom().location().clone(),
          });
        }
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
}

impl NodeVisitor<Rule> for TypeInference {
  fn visit(&mut self, rule: &Rule) {
    for pred in rule.head().iter_predicates() {
      // Check if a head predicate is a foreign predicate
      if self.foreign_predicate_type_registry.contains_predicate(&pred) {
        self.errors.push(TypeInferenceError::CannotRedefineForeignPredicate {
          pred: pred.to_string(),
          loc: rule.location().clone(),
        });
        return;
      }
    }

    // Otherwise, create a rule inference context
    let ctx = LocalTypeInferenceContext::from_rule(rule);

    // Check if context has error already
    if !ctx.errors.is_empty() {
      self.errors.extend(ctx.errors);
      return;
    }

    // First unify atom arity
    if let Err(err) = ctx.unify_atom_arities(&self.foreign_predicate_type_registry, &mut self.inferred_relation_types) {
      self.errors.push(err);
      return;
    }

    // Add the context
    self.rule_local_contexts.push(ctx);
  }
}

impl NodeVisitor<Query> for TypeInference {
  fn visit(&mut self, query: &Query) {
    // Check if the relation is a foreign predicate
    let pred = query.formatted_predicate();
    if self.foreign_predicate_type_registry.contains_predicate(&pred) {
      self.errors.push(TypeInferenceError::CannotQueryForeignPredicate {
        pred: pred.to_string(),
        loc: query.location().clone(),
      });
      return;
    }

    // Check the query
    match query {
      Query::Atom(atom) => {
        let ctx = LocalTypeInferenceContext::from_atom(atom);

        // Check if context has error already
        if !ctx.errors.is_empty() {
          self.errors.extend(ctx.errors);
          return;
        }

        // Unify atom arity
        if let Err(err) =
          ctx.unify_atom_arities(&self.foreign_predicate_type_registry, &mut self.inferred_relation_types)
        {
          self.errors.push(err);
          return;
        }

        // Add the context
        self.rule_local_contexts.push(ctx);
      }
      Query::Predicate(p) => {
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
