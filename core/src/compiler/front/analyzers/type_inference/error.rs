use crate::common::value_type::*;
use crate::compiler::front::*;

use super::*;

#[derive(Clone, Debug)]
pub enum TypeInferenceError {
  UnknownRelation {
    relation: String,
  },
  DuplicateTypeDecl {
    type_name: String,
    source_decl_loc: NodeLocation,
    duplicate_decl_loc: NodeLocation,
  },
  DuplicateRelationTypeDecl {
    predicate: String,
    source_decl_loc: NodeLocation,
    duplicate_decl_loc: NodeLocation,
  },
  UnknownADTVariant {
    predicate: String,
    loc: NodeLocation,
  },
  InvalidSubtype {
    source_type: String,
    source_type_loc: NodeLocation,
  },
  UnknownCustomType {
    type_name: String,
    loc: NodeLocation,
  },
  UnknownQueryRelationType {
    predicate: String,
    loc: NodeLocation,
  },
  UnknownFunctionType {
    function_name: String,
    loc: NodeLocation,
  },
  UnknownVariable {
    variable: String,
    loc: NodeLocation,
  },
  ArityMismatch {
    predicate: String,
    expected: usize,
    actual: usize,
    source_loc: NodeLocation,
    mismatch_loc: NodeLocation,
  },
  FunctionArityMismatch {
    function: String,
    actual: usize,
    loc: NodeLocation,
  },
  ADTVariantArityMismatch {
    variant: String,
    expected: usize,
    actual: usize,
    loc: NodeLocation,
  },
  EntityTupleArityMismatch {
    predicate: String,
    expected: usize,
    actual: usize,
    source_loc: NodeLocation,
  },
  InvalidArgIndex {
    predicate: String,
    index: usize,
    source_loc: NodeLocation,
    access_loc: NodeLocation,
  },
  InvalidForeignPredicateArgIndex {
    predicate: String,
    index: usize,
    access_loc: NodeLocation,
  },
  ConstantSetArityMismatch {
    predicate: String,
    decl_loc: NodeLocation,
    mismatch_tuple_loc: NodeLocation,
  },
  ConstantTypeMismatch {
    expected: ValueType,
    found: TypeSet,
  },
  BadEnumValueKind {
    found: &'static str,
    loc: NodeLocation,
  },
  NegativeEnumValue {
    found: i64,
    loc: NodeLocation,
  },
  CannotUnifyTypes {
    t1: TypeSet,
    t2: TypeSet,
    loc: Option<NodeLocation>,
  },
  CannotUnifyForeignPredicateArgument {
    pred: String,
    i: usize,
    expected_ty: TypeSet,
    actual_ty: TypeSet,
    loc: NodeLocation,
  },
  NoMatchingTripletRule {
    op1_ty: TypeSet,
    op2_ty: TypeSet,
    e_ty: TypeSet,
    location: NodeLocation,
  },
  CannotUnifyVariables {
    v1: String,
    t1: TypeSet,
    v2: String,
    t2: TypeSet,
    loc: NodeLocation,
  },
  CannotTypeCast {
    t1: TypeSet,
    t2: ValueType,
    loc: NodeLocation,
  },
  ConstraintNotBoolean {
    ty: TypeSet,
    loc: NodeLocation,
  },
  InvalidReduceOutput {
    op: String,
    expected: usize,
    found: usize,
    loc: NodeLocation,
  },
  InvalidReduceBindingVar {
    op: String,
    expected: usize,
    found: usize,
    loc: NodeLocation,
  },
  InvalidUniqueNumParams {
    num_output_vars: usize,
    num_binding_vars: usize,
    loc: NodeLocation,
  },
  CannotRedefineForeignPredicate {
    pred: String,
    loc: NodeLocation,
  },
  CannotQueryForeignPredicate {
    pred: String,
    loc: NodeLocation,
  },
  Internal {
    error_string: String,
  },
}

impl TypeInferenceError {
  pub fn annotate_location(&mut self, new_location: &NodeLocation) {
    match self {
      Self::CannotUnifyTypes { loc, .. } => {
        *loc = Some(new_location.clone());
      }
      _ => {}
    }
  }
}

impl FrontCompileErrorTrait for TypeInferenceError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, src: &Sources) -> String {
    match self {
      Self::UnknownRelation { relation } => {
        format!("unknown relation `{relation}`")
      }
      Self::DuplicateTypeDecl {
        type_name,
        source_decl_loc,
        duplicate_decl_loc,
      } => {
        format!(
          "duplicated type declaration found for `{}`. It is originally defined here:\n{}\nwhile we find a duplicated declaration here:\n{}",
          type_name, source_decl_loc.report(src), duplicate_decl_loc.report(src),
        )
      }
      Self::DuplicateRelationTypeDecl {
        predicate,
        source_decl_loc,
        duplicate_decl_loc,
      } => {
        format!(
          "duplicated relation type declaration found for `{}`. It is originally defined here:\n{}\nwhile we find a duplicated declaration here:\n{}",
          predicate, source_decl_loc.report(src), duplicate_decl_loc.report(src)
        )
      }
      Self::UnknownADTVariant { predicate, loc } => {
        format!(
          "unknown algebraic data type variant `{predicate}`:\n{}",
          loc.report(src)
        )
      }
      Self::InvalidSubtype {
        source_type,
        source_type_loc,
      } => {
        format!(
          "cannot create subtype from `{}`\n{}",
          source_type,
          source_type_loc.report(src)
        )
      }
      Self::UnknownCustomType { type_name, loc } => {
        format!("unknown custom type `{}`\n{}", type_name, loc.report(src))
      }
      Self::UnknownQueryRelationType { predicate, loc } => {
        format!("unknown relation `{}` used in query\n{}", predicate, loc.report(src))
      }
      Self::UnknownFunctionType { function_name, loc } => {
        format!("unknown function `{}`\n{}", function_name, loc.report(src))
      }
      Self::UnknownVariable { variable, loc } => {
        format!("unknown variable `{}` in the rule\n{}", variable, loc.report(src))
      }
      Self::ArityMismatch {
        predicate,
        expected,
        actual,
        mismatch_loc,
        ..
      } => {
        format!(
          "arity mismatch for relation `{predicate}`. Expected {expected}, found {actual}:\n{}",
          mismatch_loc.report(src)
        )
      }
      Self::FunctionArityMismatch { function, actual, loc } => {
        format!(
          "wrong number of arguments for function `{}`, found {}:\n{}",
          function,
          actual,
          loc.report(src)
        )
      }
      Self::ADTVariantArityMismatch {
        variant,
        expected,
        actual,
        loc,
      } => {
        format!(
          "arity mismatch for algebraic data type variant `{variant}`. Expected {expected}, found {actual}:\n{}",
          loc.report(src)
        )
      }
      Self::EntityTupleArityMismatch {
        predicate,
        expected,
        actual,
        source_loc,
      } => {
        format!(
          "incorrect number of arguments in entity tuple for `{predicate}`. Expected {expected}, found {actual}:\n{}",
          source_loc.report(src)
        )
      }
      Self::InvalidArgIndex {
        predicate,
        index,
        source_loc,
        access_loc,
      } => {
        format!(
          "Unexpected {}-th argument for relation `{}`. The relation type is inferred here:\n{}\nerroneous access happens here:\n{}",
          index, predicate, source_loc.report(src), access_loc.report(src)
        )
      }
      Self::InvalidForeignPredicateArgIndex {
        predicate,
        index,
        access_loc,
      } => {
        format!(
          "Unexpected {}-th argument for foreign predicate `{}`:\n{}",
          index,
          predicate,
          access_loc.report(src)
        )
      }
      Self::ConstantSetArityMismatch {
        predicate,
        mismatch_tuple_loc,
        ..
      } => {
        format!(
          "arity mismatch in constant set declaration for relation `{}`:\n{}",
          predicate,
          mismatch_tuple_loc.report(src)
        )
      }
      Self::ConstantTypeMismatch { expected, found } => {
        format!(
          "type mismatch for constant. Expected `{}`, found `{}`\n{}",
          expected,
          found,
          found.location().report(src)
        )
      }
      Self::BadEnumValueKind { found, loc } => {
        format!(
          "bad enum value. Expected unsigned integers, found `{}`\n{}",
          found,
          loc.report(src),
        )
      }
      Self::NegativeEnumValue { found, loc } => {
        format!(
          "enum value `{}` found to be negative. Expected unsigned integers\n{}",
          found,
          loc.report(src),
        )
      }
      Self::CannotUnifyTypes { t1, t2, loc } => match loc {
        Some(l) => {
          format!("cannot unify types `{}` and `{}` in\n{}\nwhere the first is inferred here\n{}\nand the second is inferred here\n{}", t1, t2, l.report(src), t1.location().report(src), t2.location().report(src))
        }
        None => {
          format!(
            "cannot unify types `{}` and `{}`, where the first is declared here\n{}\nand the second is declared here\n{}",
            t1, t2, t1.location().report(src), t2.location().report(src)
          )
        }
      },
      Self::CannotUnifyForeignPredicateArgument {
        pred,
        i,
        expected_ty,
        actual_ty,
        loc,
      } => {
        format!(
          "cannot unify the type of {}-th argument of foreign predicate `{}`, expected type `{}`, found `{}`:\n{}",
          i,
          pred,
          expected_ty,
          actual_ty,
          loc.report(src),
        )
      }
      Self::CannotUnifyVariables { v1, t1, v2, t2, loc } => {
        format!(
          "cannot unify variable types: `{}` has `{}` type, `{}` has `{}` type, but they should be unified\n{}",
          v1,
          t1,
          v2,
          t2,
          loc.report(src)
        )
      }
      Self::NoMatchingTripletRule {
        op1_ty,
        op2_ty,
        e_ty,
        location,
      } => {
        format!(
          "no matching rule found; two operands have type `{}` and `{}`, while the expression has type `{}`:\n{}",
          op1_ty,
          op2_ty,
          e_ty,
          location.report(src)
        )
      }
      Self::CannotTypeCast { t1, t2, loc } => {
        format!("cannot cast type from `{}` to `{}`\n{}", t1, t2, loc.report(src))
      }
      Self::ConstraintNotBoolean { ty, loc } => {
        format!(
          "constraint must have `bool` type, found `{}` type\n{}",
          ty,
          loc.report(src)
        )
      }
      Self::InvalidReduceOutput {
        op,
        expected,
        found,
        loc,
      } => {
        format!(
          "invalid amount of output for `{}`. Expected {}, found {}\n{}",
          op,
          expected,
          found,
          loc.report(src)
        )
      }
      Self::InvalidReduceBindingVar {
        op,
        expected,
        found,
        loc,
      } => {
        format!(
          "invalid amount of binding variables for `{}`. Expected {}, found {}\n{}",
          op,
          expected,
          found,
          loc.report(src)
        )
      }
      Self::InvalidUniqueNumParams {
        num_output_vars,
        num_binding_vars,
        loc,
      } => {
        format!(
          "expected same amount of output variables and binding variables for aggregation `unique`, but found {} output variables and {} binding variables\n{}",
          num_output_vars, num_binding_vars, loc.report(src)
        )
      }
      Self::CannotRedefineForeignPredicate { pred, loc } => {
        format!(
          "the predicate `{}` is being defined here, but it is also a foreign predicate which cannot be populated\n{}",
          pred,
          loc.report(src),
        )
      }
      Self::CannotQueryForeignPredicate { pred, loc } => {
        format!(
          "the foreign predicate `{}` cannot be queried\n{}",
          pred,
          loc.report(src),
        )
      }
      Self::Internal { error_string } => error_string.clone(),
    }
  }
}
