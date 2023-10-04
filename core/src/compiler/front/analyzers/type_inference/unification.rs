use std::collections::*;

use super::*;

use crate::common::foreign_aggregate::*;
use crate::common::value_type::*;
use crate::compiler::front::*;

type Error = FrontCompileErrorMessage;

/// The structure storing unification relationships
#[derive(Clone, Debug)]
pub enum Unification {
  /// The i-th element of a relation: arg, relation name, argument ID
  IthArgOfRelation(Loc, String, usize),

  /// V, Variable Name
  OfVariable(Loc, String),

  /// C, Type Set of C
  OfConstant(Loc, TypeSet),

  /// op1, op2, op1 + op2
  Add(Loc, Loc, Loc),

  /// op1, op2, op1 - op2
  Sub(Loc, Loc, Loc),

  /// op1, op2, op1 * op2
  Mult(Loc, Loc, Loc),

  /// op1, op2, op1 / op2
  Div(Loc, Loc, Loc),

  /// op1, op2, op1 % op2
  Mod(Loc, Loc, Loc),

  /// op1, op2, op1 == op2
  EqNeq(Loc, Loc, Loc),

  /// op1, op2, op1 && op2
  AndOrXor(Loc, Loc, Loc),

  /// op1, op2, op1 <> op2
  LtLeqGtGeq(Loc, Loc, Loc),

  /// op1, -op1
  PosNeg(Loc, Loc),

  /// op1, !op1
  Not(Loc, Loc),

  /// if X then Y else Z, X, Y, Z
  IfThenElse(Loc, Loc, Loc, Loc),

  /// op1, op1 as TY, TY
  TypeCast(Loc, Loc, Type),

  /// f, ops*, $f(ops*)
  Call(String, Vec<Loc>, Loc),

  /// var* := AGGREGATE<param*>![arg*](in_var*: ...)
  /// var*, AGGREGATE name, AGGREGATE (loc), param*, arg*, in_var*, has_exclamation_mark
  Aggregate(Vec<Loc>, String, Loc, Vec<Loc>, Vec<Loc>, Vec<Loc>, bool),

  /// C, ops*, new C(ops*)
  New(String, Vec<Loc>, Loc),
}

impl Unification {
  /// Try unifying the data types, and store the expression types to `inferred_expr_types`
  pub fn unify(
    &self,
    custom_types: &HashMap<String, (ValueType, Loc)>,
    constant_types: &HashMap<Loc, Type>,
    inferred_relation_types: &HashMap<String, (Vec<TypeSet>, Loc)>,
    function_type_registry: &FunctionTypeRegistry,
    predicate_type_registry: &PredicateTypeRegistry,
    aggregate_type_registry: &AggregateTypeRegistry,
    inferred_expr_types: &mut HashMap<Loc, TypeSet>,
  ) -> Result<(), Error> {
    match self {
      Self::IthArgOfRelation(e, p, i) => {
        if let Some(tys) = predicate_type_registry.get(p) {
          if i < &tys.len() {
            // It is a foreign predicate in the registry; we get the i-th type
            let ty = TypeSet::base(tys[*i].clone());

            // Unify the type
            match unify_ty(e, ty.clone(), inferred_expr_types) {
              Ok(_) => Ok(()),
              Err(_) => Err(Error::cannot_unify_foreign_predicate_arg(
                p.clone(),
                *i,
                ty,
                inferred_expr_types.get(e).unwrap().clone(),
                e.clone(),
              )),
            }
          } else {
            Err(Error::invalid_foreign_predicate_arg_index(
              p.clone(),
              i.clone(),
              e.clone(),
            ))
          }
        } else {
          // It is user defined predicate
          let (tys, loc) = inferred_relation_types.get(p).unwrap();
          if i < &tys.len() {
            let ty = tys[*i].clone();
            unify_ty(e, ty, inferred_expr_types).map_err(|e| e.into())?;
            Ok(())
          } else {
            Err(Error::invalid_predicate_arg_index(
              p.clone(),
              i.clone(),
              loc.clone(),
              e.clone(),
            ))
          }
        }
      }
      Self::OfVariable(_, _) => Ok(()),
      Self::OfConstant(e, ty) => {
        // Check if the constant is typed or not
        if let Some(const_decl_type) = constant_types.get(e) {
          // First try to resolve for the type name
          let t = match find_value_type(custom_types, const_decl_type) {
            Ok(base_ty) => TypeSet::BaseType(base_ty, const_decl_type.location().clone()),
            Err(err) => return Err(err),
          };

          // Unify the type name and the constant type
          let t = match ty.unify(&t) {
            Ok(t) => t,
            Err(mut err) => {
              err.annotate_location(e);
              return Err(err.into());
            }
          };

          // Update the type
          inferred_expr_types.insert(e.clone(), t);
          Ok(())
        } else {
          // If the constant is not typed, simply check the inferred expression types
          unify_ty(e, ty.clone(), inferred_expr_types).map_err(|e| e.into())?;
          Ok(())
        }
      }
      Self::Add(op1, op2, e) => {
        unify_polymorphic_binary_expression(op1, op2, e, inferred_expr_types, &ADD_TYPING_RULES)
      }
      Self::Sub(op1, op2, e) => {
        unify_polymorphic_binary_expression(op1, op2, e, inferred_expr_types, &SUB_TYPING_RULES)
      }
      Self::Mult(op1, op2, e) => {
        unify_polymorphic_binary_expression(op1, op2, e, inferred_expr_types, &MULT_TYPING_RULES)
      }
      Self::Div(op1, op2, e) => {
        unify_polymorphic_binary_expression(op1, op2, e, inferred_expr_types, &DIV_TYPING_RULES)
      }
      Self::Mod(op1, op2, e) => {
        let e_ty = inferred_expr_types
          .entry(e.clone())
          .or_insert(TypeSet::Arith(e.clone()))
          .clone();
        let op1_ty = inferred_expr_types
          .entry(op1.clone())
          .or_insert(TypeSet::Arith(op1.clone()))
          .clone();
        let op2_ty = inferred_expr_types
          .entry(op2.clone())
          .or_insert(TypeSet::Arith(op2.clone()))
          .clone();
        match op1_ty.unify(&op2_ty).and_then(|t| t.unify(&e_ty)) {
          Ok(new_ty) => {
            inferred_expr_types.insert(e.clone(), new_ty.clone());
            inferred_expr_types.insert(op1.clone(), new_ty.clone());
            inferred_expr_types.insert(op2.clone(), new_ty);
            Ok(())
          }
          Err(mut err) => {
            err.annotate_location(e);
            Err(err.into())
          }
        }
      }
      Self::EqNeq(op1, op2, e) => {
        // The type of e is boolean
        unify_boolean(e, inferred_expr_types).map_err(|e| e.into())?;

        // The two operators are of the same type
        let op_ty = TypeSet::Any(op1.clone());
        let old_op1_ty = inferred_expr_types.entry(op1.clone()).or_insert(op_ty.clone()).clone();
        let old_op2_ty = inferred_expr_types.entry(op2.clone()).or_insert(op_ty).clone();
        match old_op1_ty.unify(&old_op2_ty) {
          Ok(new_op_ty) => {
            inferred_expr_types.insert(op1.clone(), new_op_ty.clone());
            inferred_expr_types.insert(op2.clone(), new_op_ty);
            Ok(())
          }
          Err(mut err) => {
            err.annotate_location(e);
            Err(err.into())
          }
        }
      }
      Self::AndOrXor(op1, op2, e) => {
        // All e, op1, and op2 are boolean
        unify_boolean(e, inferred_expr_types).map_err(|e| e.into())?;
        unify_boolean(op1, inferred_expr_types).map_err(|e| e.into())?;
        unify_boolean(op2, inferred_expr_types).map_err(|e| e.into())?;

        Ok(())
      }
      Self::LtLeqGtGeq(op1, op2, e) => {
        unify_comparison_expression(op1, op2, e, inferred_expr_types, &COMPARE_TYPING_RULES)
      }
      Self::PosNeg(op1, e) => {
        let e_ty = inferred_expr_types
          .entry(e.clone())
          .or_insert(TypeSet::Arith(e.clone()))
          .clone();
        let op1_ty = inferred_expr_types
          .entry(op1.clone())
          .or_insert(TypeSet::Arith(op1.clone()))
          .clone();
        match e_ty.unify(&op1_ty) {
          Ok(new_ty) => {
            inferred_expr_types.insert(e.clone(), new_ty.clone());
            inferred_expr_types.insert(op1.clone(), new_ty);
            Ok(())
          }
          Err(mut err) => {
            err.annotate_location(e);
            Err(err.into())
          }
        }
      }
      Self::Not(op1, e) => {
        // e and op1 should both be boolean
        unify_boolean(e, inferred_expr_types).map_err(|e| e.into())?;
        unify_boolean(op1, inferred_expr_types).map_err(|e| e.into())?;

        Ok(())
      }
      Self::IfThenElse(e, cond, then_br, else_br) => {
        // cond should be boolean
        unify_boolean(cond, inferred_expr_types).map_err(|e| e.into())?;

        // Make sure that the expression, the then branch, and the else branch all have the same type
        let e_ty = get_or_insert_ty(e, TypeSet::Any(e.clone()), inferred_expr_types);
        let then_br_ty = get_or_insert_ty(then_br, TypeSet::Any(then_br.clone()), inferred_expr_types);
        let else_br_ty = get_or_insert_ty(else_br, TypeSet::Any(else_br.clone()), inferred_expr_types);
        match e_ty.unify(&then_br_ty).and_then(|t| t.unify(&else_br_ty)) {
          Ok(new_ty) => {
            inferred_expr_types.insert(e.clone(), new_ty.clone());
            inferred_expr_types.insert(then_br.clone(), new_ty.clone());
            inferred_expr_types.insert(else_br.clone(), new_ty);
            Ok(())
          }
          Err(mut err) => {
            err.annotate_location(e);
            Err(err.into())
          }
        }
      }
      Self::TypeCast(op1, e, ty) => {
        // Resulting type should be ty
        let ts = match find_value_type(custom_types, ty) {
          Ok(base_ty) => TypeSet::BaseType(base_ty, ty.location().clone()),
          Err(err) => return Err(err),
        };
        unify_ty(e, ts, inferred_expr_types).map_err(|e| e.into())?;

        // op1 can be any type (for now)
        unify_any(op1, inferred_expr_types).map_err(|e| e.into())?;

        Ok(())
      }
      Self::Call(function, args, e) => {
        // First check if there is such function
        if let Some(function_type) = function_type_registry.get(function) {
          // Then check if the provided number of arguments is valid
          if function_type.is_valid_num_args(args.len()) {
            // Store a collection of all the type parameters
            let mut generic_type_param_instances = HashMap::<usize, Vec<Loc>>::new();

            // Iterate through all the arguments
            for (i, arg) in args.iter().enumerate() {
              // NOTE: unwrap is ok since argument number is already checked
              let expected_arg_type = function_type.type_of_ith_argument(i).unwrap();
              match expected_arg_type {
                FunctionArgumentType::Generic(generic_type_param_id) => {
                  // Add this argument to the to-unify generic type param list
                  generic_type_param_instances
                    .entry(generic_type_param_id)
                    .or_default()
                    .push(arg.clone());
                }
                FunctionArgumentType::TypeSet(ts) => {
                  // Unify arg type for non-generic ones
                  unify_ty(arg, ts, inferred_expr_types).map_err(|e| e.into())?;
                }
              }
            }

            // Get the return type
            match &function_type.return_type {
              FunctionReturnType::Generic(generic_type_param_id) => {
                // Add the whole function expression to the to-unify generic type param list
                generic_type_param_instances
                  .entry(*generic_type_param_id)
                  .or_default()
                  .push(e.clone());
              }
              FunctionReturnType::BaseType(t) => {
                // Unify the return type with the base type
                let ts = TypeSet::base(t.clone());
                unify_ty(e, ts, inferred_expr_types).map_err(|e| e.into())?;
              }
            }

            // Unify for each generic type parameter
            //
            // There are two goals:
            // 1. All arguments of each generic type parameter should be unified with the type family
            // 2. All arguments of each generic type parameter should be the same
            for (i, generic_type_family) in function_type.generic_type_parameters.iter().enumerate() {
              // Check if there is actually instance
              if let Some(instances) = generic_type_param_instances.get(&i) {
                if instances.len() >= 1 {
                  // Keep an aggregated unified ts starting from the first instance
                  let mut agg_unified_ts =
                    unify_ty(&instances[0], generic_type_family.clone(), inferred_expr_types).map_err(|e| e.into())?;

                  // Iterate from the next instance
                  for j in 1..instances.len() {
                    // Make sure the current type conform to the generic type parameter
                    let curr_unified_ts = unify_ty(&instances[j], generic_type_family.clone(), inferred_expr_types)
                      .map_err(|e| e.into())?;

                    // Unify with the aggregated type set
                    agg_unified_ts = agg_unified_ts.unify(&curr_unified_ts).map_err(|e| e.into())?;
                  }

                  // At the end, update all instances to have the `agg_unified_ts` type
                  for instance in instances {
                    inferred_expr_types.insert(instance.clone(), agg_unified_ts.clone());
                  }
                }
              }
            }

            // No more error
            Ok(())
          } else {
            Err(Error::function_arity_mismatch(function.clone(), args.len(), e.clone()))
          }
        } else {
          Err(Error::unknown_function_type(function.clone(), e.clone()))
        }
      }
      Self::Aggregate(out_vars, agg, agg_loc, param_consts, arg_vars, in_vars, has_exclamation) => {
        if let Some(agg_type) = aggregate_type_registry.get(agg) {
          // 1. check the parameters length match
          let mut has_optional = false;
          let mut curr_param_const_id = 0;
          for (i, param_type) in agg_type.param_types.iter().enumerate() {
            match param_type {
              ParamType::Mandatory(vt) => {
                if has_optional {
                  return Err(Error::error()
                    .msg(format!("error in aggregate `{agg}`: mandatory parameter must occur before optional parameter")));
                } else if let Some(curr_param) = param_consts.get(curr_param_const_id) {
                  unify_ty(curr_param, TypeSet::base(vt.clone()), inferred_expr_types).map_err(|e| e.into())?;
                  curr_param_const_id += 1;
                } else {
                  return Err(Error::error()
                    .msg(format!("mandatory {i}-th {vt} parameter not found for aggregate `{agg}`:"))
                    .src(agg_loc.clone()))
                }
              }
              ParamType::Optional(vt) => {
                has_optional = true;
                if let Some(curr_param) = param_consts.get(curr_param_const_id) {
                  match unify_ty(curr_param, TypeSet::base(vt.clone()), inferred_expr_types) {
                    Ok(_) => {
                      curr_param_const_id += 1;
                    }
                    Err(_) => {}
                  }
                }
              }
            }
          }

          // 2. check if there is any extra parameter not scanned
          if curr_param_const_id + 1 < param_consts.len() {
            return Err(Error::error()
              .msg(format!("expected at most {} parameters, found {} parameters", agg_type.param_types.len(), param_consts.len()))
              .src(agg_loc.clone()))
          }

          // 3. check the exclamation mark
          if *has_exclamation && !agg_type.allow_exclamation_mark {
            return Err(
              Error::error()
                .msg(format!("aggregator `{agg}` does not support exclamation mark"))
                .src(agg_loc.clone()),
            );
          }

          // 4. trying to ground generics for the arg_vars or in_vars
          let mut grounded_generic_types = HashMap::new();
          ground_input_aggregate_binding_type(
            "argument",
            agg,
            agg_loc,
            &agg_type.arg_type,
            arg_vars,
            &agg_type.generics,
            &mut grounded_generic_types,
            inferred_expr_types,
          )?;
          ground_input_aggregate_binding_type(
            "input",
            agg,
            agg_loc,
            &agg_type.input_type,
            in_vars,
            &agg_type.generics,
            &mut grounded_generic_types,
            inferred_expr_types,
          )?;

          // 5. trying to unify the out_vars
          ground_output_aggregate_binding_type(
            agg,
            agg_loc,
            &agg_type.output_type,
            out_vars,
            &grounded_generic_types,
            inferred_expr_types,
          )?;

          // If passed all the tests, all good!
          Ok(())
        } else {
          Err(
            Error::error()
              .msg(format!("unknown aggregate `{agg}`"))
              .src(agg_loc.clone()),
          )
        }
      }
      Self::New(functor, args, e) => {
        let adt_variant_relation_name = format!("adt#{functor}");

        // cond should be boolean
        unify_entity(e, inferred_expr_types).map_err(|e| e.into())?;

        // Get the functor/relation
        if let Some((types, _)) = inferred_relation_types.get(&adt_variant_relation_name) {
          if args.len() == types.len() - 1 {
            for (arg, ty) in args.iter().zip(types.iter().skip(1)) {
              let arg_ty = get_or_insert_ty(arg, TypeSet::Any(arg.clone()), inferred_expr_types);
              match arg_ty.unify(ty) {
                Ok(new_ty) => {
                  inferred_expr_types.insert(arg.clone(), new_ty);
                }
                Err(mut err) => {
                  err.annotate_location(arg);
                  return Err(err.into());
                }
              }
            }
            Ok(())
          } else {
            Err(Error::adt_variant_arity_mismatch(
              functor.clone(),
              types.len() - 1,
              args.len(),
              e.clone(),
            ))
          }
        } else {
          Err(Error::unknown_adt_variant(functor.clone(), e.clone()))
        }
      }
    }
  }
}

enum AppliedRules<T> {
  None,
  One(T),
  Multiple,
}

impl<T> AppliedRules<T> {
  fn new() -> Self {
    Self::None
  }

  fn add(self, rule: T) -> Self {
    match self {
      Self::None => Self::One(rule),
      Self::One(_) => Self::Multiple,
      Self::Multiple => Self::Multiple,
    }
  }
}

fn get_or_insert_ty(e: &Loc, ty: TypeSet, inferred_expr_types: &mut HashMap<Loc, TypeSet>) -> TypeSet {
  inferred_expr_types.entry(e.clone()).or_insert(ty).clone()
}

fn unify_polymorphic_binary_expression(
  op1: &Loc,
  op2: &Loc,
  e: &Loc,
  inferred_expr_types: &mut HashMap<Loc, TypeSet>,
  rules: &[(ValueType, ValueType, ValueType)],
) -> Result<(), Error> {
  // First get the already inferred types of op1, op2, and e
  let op1_ty = unify_any(op1, inferred_expr_types).map_err(|e| e.into())?;
  let op2_ty = unify_any(op2, inferred_expr_types).map_err(|e| e.into())?;
  let e_ty = unify_any(e, inferred_expr_types).map_err(|e| e.into())?;

  // Then iterate through all the rules to see if any could be applied
  let mut applied_rules = AppliedRules::new();
  for (t1, t2, te) in rules {
    if op1_ty.contains_value_type(t1) && op2_ty.contains_value_type(t2) && e_ty.contains_value_type(te) {
      applied_rules = applied_rules.add((t1.clone(), t2.clone(), te.clone()));
    }
  }

  // Finally, check if there is any rule applied
  match applied_rules {
    AppliedRules::None => {
      // If no rule can be applied, then the type inference is failed
      Err(Error::no_matching_triplet_rule(op1_ty, op2_ty, e_ty, e.clone()))
    }
    AppliedRules::One((t1, t2, te)) => {
      // If there is exactly one rule that can be applied, then unify them with the exact types
      unify_ty(op1, TypeSet::BaseType(t1, e.clone()), inferred_expr_types).map_err(|e| e.into())?;
      unify_ty(op2, TypeSet::BaseType(t2, e.clone()), inferred_expr_types).map_err(|e| e.into())?;
      unify_ty(e, TypeSet::BaseType(te, e.clone()), inferred_expr_types).map_err(|e| e.into())?;
      Ok(())
    }
    AppliedRules::Multiple => {
      // If ther are multiple rules that can be applied, we are not sure about the exact types,
      // but the type inference is still successful
      Ok(())
    }
  }
}

fn unify_comparison_expression(
  op1: &Loc,
  op2: &Loc,
  e: &Loc,
  inferred_expr_types: &mut HashMap<Loc, TypeSet>,
  rules: &[(ValueType, ValueType)],
) -> Result<(), Error> {
  // The result should be a boolean
  let e_ty = unify_boolean(e, inferred_expr_types).map_err(|e| e.into())?;

  // First get the already inferred types of op1, op2, and e
  let op1_ty = unify_any(op1, inferred_expr_types).map_err(|e| e.into())?;
  let op2_ty = unify_any(op2, inferred_expr_types).map_err(|e| e.into())?;

  // Then iterate through all the rules to see if any could be applied
  let mut applied_rules = AppliedRules::new();
  for (t1, t2) in rules {
    if op1_ty.contains_value_type(t1) && op2_ty.contains_value_type(t2) {
      applied_rules = applied_rules.add((t1.clone(), t2.clone()));
    }
  }

  // Finally, check if there is any rule applied
  match applied_rules {
    AppliedRules::None => {
      // If no rule can be applied, then the type inference is failed
      Err(Error::no_matching_triplet_rule(op1_ty, op2_ty, e_ty, e.clone()))
    }
    AppliedRules::One((t1, t2)) => {
      // If there is exactly one rule that can be applied, then unify them with the exact types
      unify_ty(op1, TypeSet::BaseType(t1, e.clone()), inferred_expr_types).map_err(|e| e.into())?;
      unify_ty(op2, TypeSet::BaseType(t2, e.clone()), inferred_expr_types).map_err(|e| e.into())?;
      Ok(())
    }
    AppliedRules::Multiple => {
      // If ther are multiple rules that can be applied, we are not sure about the exact types,
      // but the type inference is still successful
      Ok(())
    }
  }
}

fn unify_ty(
  e: &Loc,
  ty: TypeSet,
  inferred_expr_types: &mut HashMap<Loc, TypeSet>,
) -> Result<TypeSet, CannotUnifyTypes> {
  let old_e_ty = inferred_expr_types.entry(e.clone()).or_insert(ty.clone());
  match ty.unify(old_e_ty) {
    Ok(new_e_ty) => {
      *old_e_ty = new_e_ty.clone();
      Ok(new_e_ty)
    }
    Err(mut err) => {
      err.annotate_location(e);
      Err(err)
    }
  }
}

fn unify_any(e: &Loc, inferred_expr_types: &mut HashMap<Loc, TypeSet>) -> Result<TypeSet, CannotUnifyTypes> {
  let e_ty = TypeSet::Any(e.clone());
  unify_ty(e, e_ty, inferred_expr_types)
}

fn unify_boolean(e: &Loc, inferred_expr_types: &mut HashMap<Loc, TypeSet>) -> Result<TypeSet, CannotUnifyTypes> {
  let e_ty = TypeSet::BaseType(ValueType::Bool, e.clone());
  unify_ty(e, e_ty, inferred_expr_types)
}

fn unify_entity(e: &Loc, inferred_expr_types: &mut HashMap<Loc, TypeSet>) -> Result<TypeSet, CannotUnifyTypes> {
  let e_ty = TypeSet::BaseType(ValueType::Entity, e.clone());
  unify_ty(e, e_ty, inferred_expr_types)
}

/// Given a binding type of an aggregate and the concrete variables for the aggregate, check the variable types and
/// potentially ground the generic types if they present
fn ground_input_aggregate_binding_type(
  kind: &str,
  aggregate: &str,
  aggregate_loc: &Loc,
  binding_types: &BindingTypes,
  variables: &Vec<Loc>,
  generic_type_families: &HashMap<String, GenericTypeFamily>,
  grounded_generic_types: &mut HashMap<String, Vec<TypeSet>>,
  inferred_expr_types: &mut HashMap<Loc, TypeSet>,
) -> Result<(), Error> {
  // First match on binding types
  match binding_types {
    BindingTypes::IfNotUnit { .. } => {
      // Input binding types cannot have if-not-unit expression
      Err(Error::error().msg(format!(
        "error in aggregate `{aggregate}`: cannot have if-not-unit binding type in aggregate {kind}"
      )))
    }
    BindingTypes::TupleType(elems) => {
      if elems.len() == 0 {
        // If elems.len() is 0, it means that there should be no variable for this part of aggregation.
        // We throw error if there is at least 1 variable.
        // Otherwise, the type checking is done as there is no variable that needs to be unified for type
        if variables.len() != 0 {
          Err(
            Error::error()
              .msg(format!(
                "unexpected {kind} variables in aggregate `{aggregate}`. Expected 0, found {}",
                variables.len()
              ))
              .src(aggregate_loc.clone()),
          )
        } else {
          Ok(())
        }
      } else if elems.len() == 1 {
        // If elems.len() is 1, we could have that exact element to be a generic type variable or a concrete value type
        match &elems[0] {
          BindingType::Generic(g) => {
            if let Some(grounded_type_sets) = grounded_generic_types.get(g) {
              if grounded_type_sets.len() != variables.len() {
                Err(
                  Error::error()
                    .msg(
                      format!(
                        "the generic type `{g}` in aggregate `{aggregate}` is grounded to have {} variables; however, it is unified with a set of {} variables:",
                        grounded_type_sets.len(),
                        variables.len()
                      )
                    )
                    .src(aggregate_loc.clone())
                )
              } else {
                for (grounded_type_set, variable_loc) in grounded_type_sets.iter().zip(variables.iter()) {
                  unify_ty(variable_loc, grounded_type_set.clone(), inferred_expr_types).map_err(|e| e.into())?;
                }
                Ok(())
              }
            } else if let Some(generic_type_family) = generic_type_families.get(g) {
              let grounded_type_sets = solve_generic_type(
                kind,
                aggregate,
                aggregate_loc,
                g,
                generic_type_family,
                variables,
                inferred_expr_types,
              )?;
              grounded_generic_types.insert(g.to_string(), grounded_type_sets);
              Ok(())
            } else {
              Err(Error::error().msg(format!(
                "error processing aggregate `{aggregate}`: unknown generic type parameter `{g}`"
              )))
            }
          }
          BindingType::ValueType(v) => {
            if variables.len() == 1 {
              unify_ty(&variables[0], TypeSet::base(v.clone()), inferred_expr_types).map_err(|e| e.into())?;
              Ok(())
            } else {
              // Arity mismatch
              if variables.len() == 0 {
                Err(
                  Error::error()
                    .msg(format!(
                      "expected exactly one {v} {kind} variable in aggregate `{aggregate}`, found {}",
                      variables.len()
                    ))
                    .src(aggregate_loc.clone()),
                )
              } else {
                Err(
                  Error::error()
                    .msg(format!(
                      "expected exactly one {v} {kind} variable in aggregate `{aggregate}`, found {}",
                      variables.len()
                    ))
                    .src(variables[1].clone()),
                )
              }
            }
          }
        }
      } else {
        if elems.iter().any(|e| e.is_generic()) {
          Err(Error::error().msg(format!(
            "error in aggregate `{aggregate}`: cannot have generic in the {kind} of aggregate of more than 1 elements"
          )))
        } else if elems.len() != variables.len() {
          Err(
            Error::error()
              .msg(format!(
                "expected {} {kind} variables in aggregate `{aggregate}`, found {}",
                elems.len(),
                variables.len()
              ))
              .src(aggregate_loc.clone()),
          )
        } else {
          for (elem_binding_type, variable_loc) in elems.iter().zip(variables.iter()) {
            let elem_value_type = elem_binding_type.as_value_type().unwrap(); // unwrap is ok since we have checked that no element is generic
            unify_ty(
              variable_loc,
              TypeSet::base(elem_value_type.clone()),
              inferred_expr_types,
            )
            .map_err(|e| e.into())?;
          }
          Ok(())
        }
      }
    }
  }
}

fn solve_generic_type(
  kind: &str,
  aggregate: &str,
  aggregate_loc: &Loc,
  generic_type_name: &str,
  generic_type_family: &GenericTypeFamily,
  variables: &Vec<Loc>,
  inferred_expr_types: &mut HashMap<Loc, TypeSet>,
) -> Result<Vec<TypeSet>, Error> {
  match generic_type_family {
    GenericTypeFamily::NonEmptyTuple => {
      if variables.len() == 0 {
        Err(
          Error::error()
            .msg(format!(
              "arity mismatch on aggregate `{aggregate}`. Expected non-empty {kind} variables, but found 0"
            ))
            .src(aggregate_loc.clone()),
        )
      } else {
        variables
          .iter()
          .map(|var_loc| unify_ty(var_loc, TypeSet::any(), inferred_expr_types).map_err(|e| e.into()))
          .collect::<Result<Vec<_>, _>>()
      }
    }
    GenericTypeFamily::NonEmptyTupleWithElements(elem_type_families) => {
      if elem_type_families.iter().any(|tf| !tf.is_type_family()) {
        Err(Error::error().msg(format!("error in aggregate `{aggregate}`: generic type family `{generic_type_name}` contains unsupported nested tuple")))
      } else if variables.len() != elem_type_families.len() {
        Err(
          Error::error()
            .msg(format!(
              "arity mismatch on aggregate `{aggregate}`. Expected {} {kind} variables, but found 0",
              elem_type_families.len()
            ))
            .src(aggregate_loc.clone()),
        )
      } else {
        variables
          .iter()
          .zip(elem_type_families.iter())
          .map(|(var_loc, elem_type_family)| {
            let type_family = elem_type_family.as_type_family().unwrap(); // unwrap is okay since we have checked that every elem is a base type family
            unify_ty(var_loc, TypeSet::from(type_family.clone()), inferred_expr_types).map_err(|e| e.into())
          })
          .collect::<Result<Vec<_>, _>>()
      }
    }
    GenericTypeFamily::UnitOr(child_generic_type_family) => {
      if variables.len() == 0 {
        Ok(vec![])
      } else {
        solve_generic_type(
          kind,
          aggregate,
          aggregate_loc,
          generic_type_name,
          child_generic_type_family,
          variables,
          inferred_expr_types,
        )
      }
    }
    GenericTypeFamily::TypeFamily(tf) => {
      if variables.len() != 1 {
        Err(
          Error::error()
            .msg(format!(
              "arity mismatch on aggregate `{aggregate}`. Expected 1 {kind} variables, but found 0"
            ))
            .src(aggregate_loc.clone()),
        )
      } else {
        let ts = unify_ty(&variables[0], TypeSet::from(tf.clone()), inferred_expr_types).map_err(|e| e.into())?;
        Ok(vec![ts])
      }
    }
  }
}

fn ground_output_aggregate_binding_type(
  aggregate: &str,
  aggregate_loc: &Loc,
  binding_types: &BindingTypes,
  variables: &Vec<Loc>,
  grounded_generic_types: &HashMap<String, Vec<TypeSet>>,
  inferred_expr_types: &mut HashMap<Loc, TypeSet>,
) -> Result<(), Error> {
  let expected_variable_types = solve_binding_types(aggregate, binding_types, grounded_generic_types)?;
  if expected_variable_types.len() != variables.len() {
    Err(
      Error::error()
        .msg(format!(
          "in aggregate `{aggregate}`, {} output argument(s) is expected, found {}",
          expected_variable_types.len(),
          variables.len()
        ))
        .src(aggregate_loc.clone()),
    )
  } else {
    for (expected_var_type, variable_loc) in expected_variable_types.into_iter().zip(variables.iter()) {
      unify_ty(variable_loc, expected_var_type, inferred_expr_types).map_err(|e| e.into())?;
    }
    Ok(())
  }
}

fn solve_binding_types(
  aggregate: &str,
  binding_types: &BindingTypes,
  grounded_generic_types: &HashMap<String, Vec<TypeSet>>,
) -> Result<Vec<TypeSet>, Error> {
  match binding_types {
    BindingTypes::IfNotUnit {
      generic_type,
      then_type,
      else_type,
    } => {
      if let Some(type_sets) = grounded_generic_types.get(generic_type) {
        if type_sets.len() > 0 {
          solve_binding_types(aggregate, then_type, grounded_generic_types)
        } else {
          solve_binding_types(aggregate, else_type, grounded_generic_types)
        }
      } else {
        Err(Error::error().msg(format!(
          "error grounding output type of aggregate `{aggregate}`: unknown generic type `{generic_type}`"
        )))
      }
    }
    BindingTypes::TupleType(elems) => Ok(
      elems
        .iter()
        .map(|elem| match elem {
          BindingType::Generic(g) => {
            if let Some(type_sets) = grounded_generic_types.get(g) {
              Ok(type_sets.clone())
            } else {
              Err(Error::error().msg(format!(
                "error grounding output type of aggregate `{aggregate}`: unknown generic type `{g}`"
              )))
            }
          }
          BindingType::ValueType(v) => Ok(vec![TypeSet::base(v.clone())]),
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flat_map(|es| es)
        .collect(),
    ),
  }
}
