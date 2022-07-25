use crate::common::tuple::Tuple;
use crate::common::tuple_type::TupleType;

pub enum EDBError {
  /// When there is a relation with its type stored in the EDB, and the type is a mismatch
  RelationTypeError {
    relation: String,
    expected: TupleType,
    found: TupleType,
    actual: Tuple,
  },

  /// When there is no relation stored in the EDB already, but we cannot unify the types within a set of facts in itself
  TypeMismatch {
    expected: TupleType,
    found: TupleType,
    actual: Tuple,
  },
}

impl std::fmt::Debug for EDBError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::RelationTypeError { relation, expected, found, actual } => {
        f.write_fmt(format_args!("[Relation Type Error] Expected type `{}` for relation `{}`, but found a tuple `{}` of type `{}`", expected, relation, actual, found))
      }
      Self::TypeMismatch { expected, found, actual } => {
        f.write_fmt(format_args!("[Type Mismatch] Expected type `{}`, but found a tuple `{}` of type `{}`", expected, found, actual))
      }
    }
  }
}
