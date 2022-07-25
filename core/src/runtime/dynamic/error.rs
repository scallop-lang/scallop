use super::io::IOError;
use crate::common::tuple_type::TupleType;

#[derive(Clone, Debug)]
pub enum RuntimeError {
  IO(IOError),
  UnknownRelation(String),
  TypeError(String, TupleType),
}

impl std::fmt::Display for RuntimeError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::IO(e) => std::fmt::Display::fmt(e, f),
      Self::UnknownRelation(r) => f.write_fmt(format_args!("Unknown relation `{}`", r)),
      Self::TypeError(tup, ty) => f.write_fmt(format_args!(
        "Type mismatch in tuple `{}` against type `{}`",
        tup, ty
      )),
    }
  }
}

impl From<IOError> for RuntimeError {
  fn from(e: IOError) -> Self {
    Self::IO(e)
  }
}
