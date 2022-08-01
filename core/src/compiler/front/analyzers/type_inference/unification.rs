use std::collections::*;

use super::*;
use crate::common::value_type::*;
use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub enum Unification {
  IthArgOfRelation(Loc, String, usize),
  OfVariable(Loc, String),
  OfConstant(Loc, TypeSet),
  AddSubMulDivMod(Loc, Loc, Loc), // op1, op2, op1 X op2
  EqNeq(Loc, Loc, Loc),           // op1, op2, op1 == op2
  AndOrXor(Loc, Loc, Loc),        // op1, op2, op1 && op2
  LtLeqGtGeq(Loc, Loc, Loc),      // op1, op2, op1 <> op2
  PosNeg(Loc, Loc),               // op1, -op1
  Not(Loc, Loc),                  // op1, !op1
  IfThenElse(Loc, Loc, Loc, Loc), // if X then Y else Z, X, Y, Z
  TypeCast(Loc, Loc, Type),       // op1, op1 as TY, TY
}

impl Unification {
  pub fn unify(
    &self,
    custom_types: &HashMap<String, (ValueType, Loc)>,
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
        unify_ty(e, ty.clone(), inferred_expr_types)?;
        Ok(())
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

        let e_ty = inferred_expr_types
          .entry(e.clone())
          .or_insert(TypeSet::Any(e.clone()))
          .clone();
        let then_br_ty = inferred_expr_types
          .entry(then_br.clone())
          .or_insert(TypeSet::Any(then_br.clone()))
          .clone();
        let else_br_ty = inferred_expr_types
          .entry(else_br.clone())
          .or_insert(TypeSet::Any(else_br.clone()))
          .clone();
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

fn unify_arith(e: &Loc, inferred_expr_types: &mut HashMap<Loc, TypeSet>) -> Result<TypeSet, TypeInferenceError> {
  let e_ty = TypeSet::Arith(e.clone());
  unify_ty(e, e_ty, inferred_expr_types)
}

fn unify_boolean(e: &Loc, inferred_expr_types: &mut HashMap<Loc, TypeSet>) -> Result<TypeSet, TypeInferenceError> {
  let e_ty = TypeSet::BaseType(ValueType::Bool, e.clone());
  unify_ty(e, e_ty, inferred_expr_types)
}
