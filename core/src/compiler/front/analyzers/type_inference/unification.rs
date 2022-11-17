use std::collections::*;

use super::*;
use crate::common::value_type::*;
use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub enum Unification {
  IthArgOfRelation(Loc, String, usize),
  OfVariable(Loc, String),
  OfConstant(Loc, TypeSet),
  AddSubMulDivMod(Loc, Loc, Loc),                          // op1, op2, op1 X op2
  EqNeq(Loc, Loc, Loc),                                    // op1, op2, op1 == op2
  AndOrXor(Loc, Loc, Loc),                                 // op1, op2, op1 && op2
  LtLeqGtGeq(Loc, Loc, Loc),                               // op1, op2, op1 <> op2
  PosNeg(Loc, Loc),                                        // op1, -op1
  Not(Loc, Loc),                                           // op1, !op1
  IfThenElse(Loc, Loc, Loc, Loc),                          // if X then Y else Z, X, Y, Z
  TypeCast(Loc, Loc, Type),                                // op1, op1 as TY, TY
  Call(crate::common::functions::Function, Vec<Loc>, Loc), // f, ops*, $f(ops*)
}

impl Unification {
  pub fn unify(
    &self,
    custom_types: &HashMap<String, (ValueType, Loc)>,
    constant_types: &HashMap<Loc, Type>,
    inferred_relation_types: &HashMap<String, (Vec<TypeSet>, Loc)>,
    inferred_expr_types: &mut HashMap<Loc, TypeSet>,
  ) -> Result<(), TypeInferenceError> {
    match self {
      Self::IthArgOfRelation(e, p, i) => {
        let (tys, loc) = inferred_relation_types.get(p).unwrap();
        if i < &tys.len() {
          let ty = tys[(*i)].clone();
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
      Self::AddSubMulDivMod(op1, op2, e) => {
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
        // e should be boolean
        unify_boolean(e, inferred_expr_types)?;

        // op1 and op2 are numeric
        let t1 = unify_arith(op1, inferred_expr_types)?;
        let t2 = unify_arith(op2, inferred_expr_types)?;

        // op1 and op2 are of the same type
        match t1.unify(&t2) {
          Ok(new_ty) => {
            inferred_expr_types.insert(op1.clone(), new_ty.clone());
            inferred_expr_types.insert(op2.clone(), new_ty.clone());
            Ok(())
          }
          Err(mut err) => {
            err.annotate_location(e);
            Err(err)
          }
        }
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
        use crate::common::functions::Function;
        match function {
          Function::Abs => {
            if args.len() == 1 {
              let arg_ty = unify_arith(&args[0], inferred_expr_types)?;
              let e_ty = unify_arith(e, inferred_expr_types)?;
              match arg_ty.unify(&e_ty) {
                Ok(new_ty) => {
                  inferred_expr_types.insert(e.clone(), new_ty.clone());
                  inferred_expr_types.insert(args[0].clone(), new_ty.clone());
                  Ok(())
                }
                Err(mut err) => {
                  err.annotate_location(e);
                  Err(err)
                }
              }
            } else {
              Err(TypeInferenceError::FunctionArityMismatch {
                function: "abs".to_string(),
                actual: args.len(),
                loc: e.clone(),
              })
            }
          }
          Function::Hash
          | Function::StringConcat
          | Function::StringLength
          | Function::Substring
          | Function::StringCharAt => {
            // Resulting type should be function return type
            let ret_ty = TypeSet::function_return_type(function);
            unify_ty(e, ret_ty, inferred_expr_types)?;

            // Args should be of their respective type
            if function.is_acceptable_num_args(args.len()) {
              for (i, arg) in args.iter().enumerate() {
                let ts = TypeSet::function_argument_type(function, i);
                unify_ty(arg, ts, inferred_expr_types)?;
              }
            } else {
              return Err(TypeInferenceError::FunctionArityMismatch {
                function: format!("{}", function),
                actual: args.len(),
                loc: e.clone(),
              });
            }

            Ok(())
          }
        }
      }
    }
  }
}

fn get_or_insert_ty(e: &Loc, ty: TypeSet, inferred_expr_types: &mut HashMap<Loc, TypeSet>) -> TypeSet {
  inferred_expr_types.entry(e.clone()).or_insert(ty).clone()
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

fn unify_arith(e: &Loc, inferred_expr_types: &mut HashMap<Loc, TypeSet>) -> Result<TypeSet, TypeInferenceError> {
  let e_ty = TypeSet::Arith(e.clone());
  unify_ty(e, e_ty, inferred_expr_types)
}

fn unify_boolean(e: &Loc, inferred_expr_types: &mut HashMap<Loc, TypeSet>) -> Result<TypeSet, TypeInferenceError> {
  let e_ty = TypeSet::BaseType(ValueType::Bool, e.clone());
  unify_ty(e, e_ty, inferred_expr_types)
}
