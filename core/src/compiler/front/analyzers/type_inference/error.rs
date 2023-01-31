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
  UnknownFunctionType {
    function_name: String,
    loc: AstNodeLocation,
  },
  UnknownVariable {
    variable: String,
    loc: AstNodeLocation,
  },
  ArityMismatch {
    predicate: String,
    expected: usize,
    actual: usize,
    source_loc: AstNodeLocation,
    mismatch_loc: AstNodeLocation,
  },
  FunctionArityMismatch {
    function: String,
    actual: usize,
    loc: AstNodeLocation,
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

impl TypeInferenceError {
  pub fn annotate_location(&mut self, new_location: &AstNodeLocation) {
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
          "arity mismatch for relation `{}`. Expected {}, found {}:\n{}",
          predicate,
          expected,
          actual,
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
      Self::InvalidArgIndex {
        predicate,
        index,
        source_loc,
        access_loc,
      } => {
        format!(
          "invalid `{}`-th argument for relation `{}`. The relation type is inferred here:\n{}\nerroneous access happens here:\n{}",
          index, predicate, source_loc.report(src), access_loc.report(src)
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
    }
  }
}
