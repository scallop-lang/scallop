use std::collections::*;

use super::*;

use crate::common::value_type::*;
use crate::compiler::front::*;

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
    inferred_expr_types: &mut HashMap<Loc, TypeSet>,
  ) -> Result<(), TypeInferenceError> {
    match self {
      Self::IthArgOfRelation(e, p, i) => {
        if let Some(tys) = predicate_type_registry.get(p) {
          if i < &tys.len() {
            // It is a foreign predicate in the registry; we get the i-th type
            let ty = TypeSet::base(tys[*i].clone());

            // Unify the type
            match unify_ty(e, ty.clone(), inferred_expr_types) {
              Ok(_) => Ok(()),
              Err(_) => Err(TypeInferenceError::CannotUnifyForeignPredicateArgument {
                pred: p.clone(),
                i: *i,
                expected_ty: ty,
                actual_ty: inferred_expr_types.get(e).unwrap().clone(),
                loc: e.clone(),
              }),
            }
          } else {
            Err(TypeInferenceError::InvalidForeignPredicateArgIndex {
              predicate: p.clone(),
              index: i.clone(),
              access_loc: e.clone(),
            })
          }
        } else {
          // It is user defined predicate
          let (tys, loc) = inferred_relation_types.get(p).unwrap();
          if i < &tys.len() {
            let ty = tys[*i].clone();
            unify_ty(e, ty, inferred_expr_types)?;
            Ok(())
          } else {
            Err(TypeInferenceError::InvalidArgIndex {
              predicate: p.clone(),
              index: i.clone(),
              source_loc: loc.clone(),
              access_loc: e.clone(),
            })
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
              return Err(err);
            }
          };

          // Update the type
          inferred_expr_types.insert(e.clone(), t);
          Ok(())
        } else {
          // If the constant is not typed, simply check the inferred expression types
          unify_ty(e, ty.clone(), inferred_expr_types)?;
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
            Err(err)
          }
        }
      }
      Self::EqNeq(op1, op2, e) => {
        // The type of e is boolean
        unify_boolean(e, inferred_expr_types)?;

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
            Err(err)
          }
        }
      }
      Self::AndOrXor(op1, op2, e) => {
        // All e, op1, and op2 are boolean
        unify_boolean(e, inferred_expr_types)?;
        unify_boolean(op1, inferred_expr_types)?;
        unify_boolean(op2, inferred_expr_types)?;

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
            Err(err)
          }
        }
      }
      Self::Not(op1, e) => {
        // e and op1 should both be boolean
        unify_boolean(e, inferred_expr_types)?;
        unify_boolean(op1, inferred_expr_types)?;

        Ok(())
      }
      Self::IfThenElse(e, cond, then_br, else_br) => {
        // cond should be boolean
        unify_boolean(cond, inferred_expr_types)?;

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
            Err(err)
          }
        }
      }
      Self::TypeCast(op1, e, ty) => {
        // Resulting type should be ty
        let ts = match find_value_type(custom_types, ty) {
          Ok(base_ty) => TypeSet::BaseType(base_ty, ty.location().clone()),
          Err(err) => return Err(err),
        };
        unify_ty(e, ts, inferred_expr_types)?;

        // op1 can be any type (for now)
        unify_any(op1, inferred_expr_types)?;

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
                  unify_ty(arg, ts, inferred_expr_types)?;
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
                unify_ty(e, ts, inferred_expr_types)?;
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
                  let mut agg_unified_ts = unify_ty(&instances[0], generic_type_family.clone(), inferred_expr_types)?;

                  // Iterate from the next instance
                  for j in 1..instances.len() {
                    // Make sure the current type conform to the generic type parameter
                    let curr_unified_ts = unify_ty(&instances[j], generic_type_family.clone(), inferred_expr_types)?;

                    // Unify with the aggregated type set
                    agg_unified_ts = agg_unified_ts.unify(&curr_unified_ts)?;
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
            Err(TypeInferenceError::FunctionArityMismatch {
              function: function.clone(),
              actual: args.len(),
              loc: e.clone(),
            })
          }
        } else {
          Err(TypeInferenceError::UnknownFunctionType {
            function_name: function.clone(),
            loc: e.clone(),
          })
        }
      }
      Self::New(functor, args, e) => {
        let adt_variant_relation_name = format!("adt#{functor}");

        // cond should be boolean
        unify_entity(e, inferred_expr_types)?;

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
                  return Err(err);
                }
              }
            }
            Ok(())
          } else {
            Err(TypeInferenceError::ADTVariantArityMismatch {
              variant: functor.clone(),
              expected: types.len() - 1,
              actual: args.len(),
              loc: e.clone(),
            })
          }
        } else {
          Err(TypeInferenceError::UnknownADTVariant {
            predicate: functor.clone(),
            loc: e.clone(),
          })
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
) -> Result<(), TypeInferenceError> {
  // First get the already inferred types of op1, op2, and e
  let op1_ty = unify_any(op1, inferred_expr_types)?;
  let op2_ty = unify_any(op2, inferred_expr_types)?;
  let e_ty = unify_any(e, inferred_expr_types)?;

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
      Err(TypeInferenceError::NoMatchingTripletRule {
        op1_ty,
        op2_ty,
        e_ty,
        location: e.clone(),
      })
    }
    AppliedRules::One((t1, t2, te)) => {
      // If there is exactly one rule that can be applied, then unify them with the exact types
      unify_ty(op1, TypeSet::BaseType(t1, e.clone()), inferred_expr_types)?;
      unify_ty(op2, TypeSet::BaseType(t2, e.clone()), inferred_expr_types)?;
      unify_ty(e, TypeSet::BaseType(te, e.clone()), inferred_expr_types)?;
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
) -> Result<(), TypeInferenceError> {
  // The result should be a boolean
  let e_ty = unify_boolean(e, inferred_expr_types)?;

  // First get the already inferred types of op1, op2, and e
  let op1_ty = unify_any(op1, inferred_expr_types)?;
  let op2_ty = unify_any(op2, inferred_expr_types)?;

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
      Err(TypeInferenceError::NoMatchingTripletRule {
        op1_ty,
        op2_ty,
        e_ty,
        location: e.clone(),
      })
    }
    AppliedRules::One((t1, t2)) => {
      // If there is exactly one rule that can be applied, then unify them with the exact types
      unify_ty(op1, TypeSet::BaseType(t1, e.clone()), inferred_expr_types)?;
      unify_ty(op2, TypeSet::BaseType(t2, e.clone()), inferred_expr_types)?;
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
) -> Result<TypeSet, TypeInferenceError> {
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

fn unify_any(e: &Loc, inferred_expr_types: &mut HashMap<Loc, TypeSet>) -> Result<TypeSet, TypeInferenceError> {
  let e_ty = TypeSet::Any(e.clone());
  unify_ty(e, e_ty, inferred_expr_types)
}

fn unify_boolean(e: &Loc, inferred_expr_types: &mut HashMap<Loc, TypeSet>) -> Result<TypeSet, TypeInferenceError> {
  let e_ty = TypeSet::BaseType(ValueType::Bool, e.clone());
  unify_ty(e, e_ty, inferred_expr_types)
}

fn unify_entity(e: &Loc, inferred_expr_types: &mut HashMap<Loc, TypeSet>) -> Result<TypeSet, TypeInferenceError> {
  let e_ty = TypeSet::BaseType(ValueType::Entity, e.clone());
  unify_ty(e, e_ty, inferred_expr_types)
}
