use crate::common::value_type::*;
use crate::compiler::front::*;

use super::*;

#[derive(Clone, Debug)]
pub enum TypeInferenceError {
  DuplicateTypeDecl {
    type_name: String,
    source_decl_loc: AstNodeLocation,
    duplicate_decl_loc: AstNodeLocation,
  },
  DuplicateRelationTypeDecl {
    predicate: String,
    source_decl_loc: AstNodeLocation,
    duplicate_decl_loc: AstNodeLocation,
  },
  InvalidSubtype {
    source_type: TypeNode,
    source_type_loc: AstNodeLocation,
  },
  UnknownCustomType {
    type_name: String,
    loc: AstNodeLocation,
  },
  UnknownQueryRelationType {
    predicate: String,
    loc: AstNodeLocation,
  },
  ArityMismatch {
    predicate: String,
    expected: usize,
    actual: usize,
    source_loc: AstNodeLocation,
    mismatch_loc: AstNodeLocation,
  },
  InvalidArgIndex {
    predicate: String,
    index: usize,
    source_loc: AstNodeLocation,
    access_loc: AstNodeLocation,
  },
  ConstantSetArityMismatch {
    predicate: String,
    decl_loc: AstNodeLocation,
    mismatch_tuple_loc: AstNodeLocation,
  },
  NonConstantInFactDecl {
    expr_loc: AstNodeLocation,
  },
  ConstantTypeMismatch {
    expected: ValueType,
    found: TypeSet,
  },
  CannotUnifyTypes {
    t1: TypeSet,
    t2: TypeSet,
    loc: Option<AstNodeLocation>,
  },
  CannotUnifyVariables {
    v1: String,
    t1: TypeSet,
    v2: String,
    t2: TypeSet,
    loc: AstNodeLocation,
  },
  CannotTypeCast {
    t1: TypeSet,
    t2: ValueType,
    loc: AstNodeLocation,
  },
  ConstraintNotBoolean {
    ty: TypeSet,
    loc: AstNodeLocation,
  },
  InvalidReduceOutput {
    op: String,
    expected: usize,
    found: usize,
    loc: AstNodeLocation,
  },
  InvalidReduceBindingVar {
    op: String,
    expected: usize,
    found: usize,
    loc: AstNodeLocation,
  },
  InvalidUniqueNumParams {
    num_output_vars: usize,
    num_binding_vars: usize,
    loc: AstNodeLocation,
  },
}

impl From<TypeInferenceError> for FrontCompileError {
  fn from(e: TypeInferenceError) -> Self {
    Self::TypeInferenceError(e)
  }
}

impl TypeInferenceError {
  pub fn annotate_location(&mut self, new_location: &AstNodeLocation) {
    match self {
      Self::CannotUnifyTypes { loc, .. } => {
        *loc = Some(new_location.clone());
      }
      _ => {}
    }
  }

  pub fn report(&self, src: &Sources) {
    match self {
      Self::DuplicateTypeDecl {
        type_name,
        source_decl_loc,
        duplicate_decl_loc,
      } => {
        println!(
          "Duplicate type declaration found for `{}`. It is originally defined here:",
          type_name
        );
        source_decl_loc.report(src);
        println!("while we find a duplicated declaration here:");
        duplicate_decl_loc.report(src);
      }
      Self::DuplicateRelationTypeDecl {
        predicate,
        source_decl_loc,
        duplicate_decl_loc,
      } => {
        println!(
          "Duplicate relation type declaration found for `{}`. It is originally defined here:",
          predicate
        );
        source_decl_loc.report(src);
        println!("while we find a duplicated declaration here:");
        duplicate_decl_loc.report(src);
      }
      Self::InvalidSubtype {
        source_type,
        source_type_loc,
      } => {
        println!("Cannot create subtype from `{}`", source_type);
        source_type_loc.report(src);
      }
      Self::UnknownCustomType { type_name, loc } => {
        println!("Unknown custom type `{}`", type_name);
        loc.report(src);
      }
      Self::UnknownQueryRelationType { predicate, loc } => {
        println!("Unknown relation `{}` used in query", predicate);
        loc.report(src);
      }
      Self::ArityMismatch {
        predicate,
        expected,
        actual,
        mismatch_loc,
        ..
      } => {
        println!(
          "Arity mismatch for relation `{}`. Expected {}, found {}:",
          predicate, expected, actual
        );
        mismatch_loc.report(src);
      }
      Self::InvalidArgIndex {
        predicate,
        index,
        source_loc,
        access_loc,
      } => {
        println!(
          "Invalid `{}`-th argument for relation `{}`. The relation type is inferred here:",
          index, predicate,
        );
        source_loc.report(src);
        println!("erroneous access happens here:");
        access_loc.report(src);
      }
      Self::ConstantSetArityMismatch {
        predicate,
        mismatch_tuple_loc,
        ..
      } => {
        println!(
          "Arity mismatch in constant set declaration for relation `{}`:",
          predicate
        );
        mismatch_tuple_loc.report(src);
      }
      Self::NonConstantInFactDecl { expr_loc } => {
        println!("Found non-constant in fact declaration:");
        expr_loc.report(src);
      }
      Self::ConstantTypeMismatch { expected, found } => {
        println!("Type mismatch for constant. Expected `{}`, found `{}`", expected, found);
        found.location().report(src);
      }
      Self::CannotUnifyTypes { t1, t2, loc } => match loc {
        Some(l) => {
          println!("Cannot unify types `{}` and `{}` in", t1, t2);
          l.report(src);
          println!("where the first is inferred here");
          t1.location().report(src);
          println!("and the second is inferred here");
          t2.location().report(src);
        }
        None => {
          println!(
            "Cannot unify types `{}` and `{}`, where the first is declared here",
            t1, t2
          );
          t1.location().report(src);
          println!("and the second is declared here");
          t2.location().report(src);
        }
      },
      Self::CannotUnifyVariables { v1, t1, v2, t2, loc } => {
        println!(
          "Cannot unify variable types: `{}` has `{}` type, `{}` has `{}` type, but they should be unified",
          v1, t1, v2, t2
        );
        loc.report(src);
      }
      Self::CannotTypeCast { t1, t2, loc } => {
        println!("Cannot cast type from `{}` to `{}`", t1, t2);
        loc.report(src);
      }
      Self::ConstraintNotBoolean { ty, loc } => {
        println!("Constraint must have `bool` type, found `{}` type", ty);
        loc.report(src);
      }
      Self::InvalidReduceOutput {
        op,
        expected,
        found,
        loc,
      } => {
        println!(
          "Invalid amount of output for `{}`. Expected {}, found {}",
          op, expected, found
        );
        loc.report(src);
      }
      Self::InvalidReduceBindingVar {
        op,
        expected,
        found,
        loc,
      } => {
        println!(
          "Invalid amount of binding variables for `{}`. Expected {}, found {}",
          op, expected, found
        );
        loc.report(src);
      }
      Self::InvalidUniqueNumParams {
        num_output_vars,
        num_binding_vars,
        loc,
      } => {
        println!(
          "Expected same amount of output variables and binding variables for aggregation `unique`, but found {} output variables and {} binding variables",
          num_output_vars, num_binding_vars
        );
        loc.report(src);
      }
    }
  }
}
