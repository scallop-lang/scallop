use crate::common::tuple::*;
use crate::common::tuple_type::*;
use crate::runtime::error::*;

#[derive(Clone, Debug)]
pub enum DatabaseError {
  TypeError {
    relation: String,
    relation_type: TupleType,
    tuple: Tuple,
  },
  UnknownRelation {
    relation: String,
  },
  NewProgramFacts {
    relation: String,
  },
  IO(IOError),
}

impl std::fmt::Display for DatabaseError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::TypeError {
        relation,
        relation_type,
        tuple,
      } => f.write_str(&format!(
        "Type mismatch on tuple `{}` for relation `{}`. Expected tuple type `{}`",
        tuple, relation, relation_type
      )),
      Self::UnknownRelation { relation } => f.write_str(&format!("Unknown relation `{}`", relation)),
      Self::NewProgramFacts { relation } => f.write_str(&format!(
        "New facts in program declared for relation `{}`; cannot incrementally compute",
        relation
      )),
      Self::IO(error) => error.fmt(f),
    }
  }
}
