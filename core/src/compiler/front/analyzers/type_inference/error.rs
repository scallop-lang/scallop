use crate::common::value_type::*;
use crate::compiler::front::*;

use super::*;

impl FrontCompileErrorMessage {
  pub fn unknown_relation(relation: String) -> Self {
    Self::error().msg(format!("unknown relation `{relation}`"))
  }

  pub fn duplicate_type_decl(type_name: String, source_decl_loc: Loc, duplicate_decl_loc: Loc) -> Self {
    Self::error()
      .msg(format!(
        "duplicated type declaration found for `{type_name}`. It is originally defined here:"
      ))
      .src(source_decl_loc)
      .msg("while we find a duplicated declaration here:")
      .src(duplicate_decl_loc)
  }

  pub fn duplicate_relation_type_decl(predicate: String, source_decl_loc: Loc, duplicate_decl_loc: Loc) -> Self {
    Self::error()
      .msg(format!(
        "duplicated relation type declaration found for `{predicate}`. It is originally defined here:"
      ))
      .src(source_decl_loc)
      .msg("while we find a duplicated declaration here:")
      .src(duplicate_decl_loc)
  }

  pub fn unknown_adt_variant(var: String, loc: Loc) -> Self {
    Self::error()
      .msg(format!("unknown algebraic data type variant `{var}`:"))
      .src(loc)
  }

  pub fn invalid_subtype(source_type: String, loc: Loc) -> Self {
    Self::error()
      .msg(format!("cannot create subtype from `{source_type}`"))
      .src(loc)
  }

  pub fn unknown_custom_type(type_name: String, loc: Loc) -> Self {
    Self::error().msg(format!("unknown custom type `{type_name}`")).src(loc)
  }

  pub fn unknown_query_relation_type(predicate: String, loc: Loc) -> Self {
    Self::error()
      .msg(format!("unknown relation `{predicate}` used in query"))
      .src(loc)
  }

  pub fn unknown_function_type(func_name: String, loc: Loc) -> Self {
    Self::error().msg(format!("unknown function `{func_name}`")).src(loc)
  }

  pub fn unknown_variable(var: String, loc: Loc) -> Self {
    Self::error()
      .msg(format!("unknown variable `{var}` in the rule"))
      .src(loc)
  }

  pub fn unknown_aggregate(agg: String, loc: Loc) -> Self {
    Self::error().msg(format!("unknown aggregate `{agg}`")).src(loc)
  }

  pub fn arity_mismatch(pred: String, expected: usize, actual: usize, mismatch_loc: Loc) -> Self {
    Self::error()
      .msg(format!(
        "arity mismatch for relation `{pred}`. Expected {expected}, found {actual}:"
      ))
      .src(mismatch_loc)
  }

  pub fn function_arity_mismatch(func: String, actual: usize, loc: Loc) -> Self {
    Self::error()
      .msg(format!(
        "bad number of arguments for function `{func}`, found {actual}:"
      ))
      .src(loc)
  }

  pub fn adt_variant_arity_mismatch(var: String, expected: usize, actual: usize, loc: Loc) -> Self {
    Self::error()
      .msg(format!(
        "arity mismatch for algebraic data type variant `{var}`. Expected {expected}, found {actual}:"
      ))
      .src(loc)
  }

  pub fn entity_tuple_arity_mismatch(pred: String, expected: usize, actual: usize, loc: Loc) -> Self {
    Self::error()
      .msg(format!(
        "incorrect number of arguments in entity tuple for `{pred}`. Expected {expected}, found {actual}:"
      ))
      .src(loc)
  }

  pub fn invalid_predicate_arg_index(pred: String, index: usize, source_loc: Loc, access_loc: Loc) -> Self {
    Self::error()
      .msg(format!(
        "Unexpected {index}-th argument for relation `{pred}`. The relation type is inferred here:"
      ))
      .src(source_loc)
      .msg("erroneous access happens here:")
      .src(access_loc)
  }

  pub fn invalid_foreign_predicate_arg_index(pred: String, index: usize, access_loc: Loc) -> Self {
    Self::error()
      .msg(format!(
        "Unexpected {index}-th argument for foreign predicate `{pred}`:"
      ))
      .src(access_loc)
  }

  pub fn constant_set_arity_mismatch(predicate: String, mismatch_loc: Loc) -> Self {
    Self::error()
      .msg(format!("arity mismatch in set for relation `{predicate}`:"))
      .src(mismatch_loc)
  }

  pub fn constant_type_mismatch(expected: ValueType, found: TypeSet) -> Self {
    Self::error()
      .msg(format!(
        "type mismatch for constant. Expected `{expected}`, found `{found}`"
      ))
      .src(found.location().clone())
  }

  pub fn bad_enum_value_kind(found: &'static str, loc: Loc) -> Self {
    Self::error()
      .msg(format!("bad enum value. Expected unsigned integers, found `{found}`"))
      .src(loc)
  }

  pub fn negative_enum_value(found: i64, loc: Loc) -> Self {
    Self::error()
      .msg(format!(
        "enum value `{found}` found to be negative. Expected unsigned integers"
      ))
      .src(loc)
  }

  pub fn cannot_unify_foreign_predicate_arg(
    pred: String,
    i: usize,
    expected_ty: TypeSet,
    actual_ty: TypeSet,
    loc: Loc,
  ) -> Self {
    Self::error()
      .msg(format!("cannot unify the type of {i}-th argument of foreign predicate `{pred}`, expected type `{expected_ty}`, found `{actual_ty}`:"))
      .src(loc)
  }

  pub fn cannot_unify_variables(v1: String, t1: TypeSet, v2: String, t2: TypeSet, loc: Loc) -> Self {
    Self::error()
      .msg(format!(
        "cannot unify variable types: `{v1}` has `{t1}` type, `{v2}` has `{t2}` type, but they should be unified:"
      ))
      .src(loc)
  }

  pub fn no_matching_triplet_rule(op1_ty: TypeSet, op2_ty: TypeSet, e_ty: TypeSet, loc: Loc) -> Self {
    Self::error()
      .msg(format!("no matching rule found; two operands have type `{op1_ty}` and `{op2_ty}`, while the expression has type `{e_ty}`:"))
      .src(loc)
  }

  pub fn cannot_type_cast(t1: TypeSet, t2: ValueType, loc: Loc) -> Self {
    Self::error()
      .msg(format!("cannot cast type from `{t1}` to `{t2}`"))
      .src(loc)
  }

  pub fn constraint_not_boolean(ty: TypeSet, loc: Loc) -> Self {
    Self::error()
      .msg(format!("constraint must have `bool` type, but found `{ty}` type"))
      .src(loc)
  }

  pub fn cannot_redefine_foreign_predicate(pred: String, loc: Loc) -> Self {
    Self::error()
      .msg(format!(
        "the predicate `{pred}` is being defined here, but it is also a foreign predicate which cannot be populated"
      ))
      .src(loc)
  }

  pub fn cannot_query_foreign_predicate(pred: String, loc: Loc) -> Self {
    Self::error()
      .msg(format!("the foreign predicate `{pred}` cannot be queried:"))
      .src(loc)
  }
}

pub struct CannotUnifyTypes {
  pub t1: TypeSet,
  pub t2: TypeSet,
  pub loc: Option<Loc>,
}

impl Into<FrontCompileErrorMessage> for CannotUnifyTypes {
  fn into(self) -> FrontCompileErrorMessage {
    if let Some(l) = self.loc {
      FrontCompileErrorMessage::error()
        .msg(format!("cannot unify types `{}` and `{}` in", &self.t1, &self.t2))
        .src(l)
        .msg("where the first is inferred here")
        .src(self.t1.location().clone())
        .msg("and the second is inferred here")
        .src(self.t2.location().clone())
    } else {
      FrontCompileErrorMessage::error()
        .msg(format!(
          "cannot unify types `{}` and `{}`, where the first is declared here",
          &self.t1, &self.t2
        ))
        .src(self.t1.location().clone())
        .msg("and the second is declared here")
        .src(self.t2.location().clone())
    }
  }
}

impl CannotUnifyTypes {
  pub fn annotate_location(&mut self, new_location: &NodeLocation) {
    self.loc = Some(new_location.clone());
  }
}
