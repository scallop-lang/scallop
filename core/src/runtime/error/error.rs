use super::io::IOError;
use crate::common::foreign_function::ForeignFunctionError;
use crate::runtime::database::DatabaseError;

#[derive(Clone, Debug)]
pub enum RuntimeError {
  IO(IOError),
  ForeignFunction(ForeignFunctionError),
  Database(DatabaseError),
}

impl std::fmt::Display for RuntimeError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::IO(e) => e.fmt(f),
      Self::ForeignFunction(e) => e.fmt(f),
      Self::Database(e) => e.fmt(f),
    }
  }
}

impl From<IOError> for RuntimeError {
  fn from(e: IOError) -> Self {
    Self::IO(e)
  }
}

impl From<DatabaseError> for RuntimeError {
  fn from(e: DatabaseError) -> Self {
    Self::Database(e)
  }
}
