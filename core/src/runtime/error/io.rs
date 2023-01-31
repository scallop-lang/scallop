use std::path::PathBuf;

use crate::common::tuple_type::TupleType;
use crate::common::value_type::ValueParseError;

#[derive(Clone, Debug)]
pub enum IOError {
  CannotOpenFile { file_path: PathBuf, error: String },
  CannotReadFile { error: String },
  CannotParseCSV { error: String },
  InvalidType { types: TupleType },
  ValueParseError { error: ValueParseError },
  CannotParseProbability { value: String },
  ArityMismatch { expected: usize, found: usize },
  CannotWriteRecord { error: String },
}

impl std::fmt::Display for IOError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::CannotOpenFile { file_path, error } => f.write_fmt(format_args!(
        "IO: Cannot open file `{}`: {}",
        file_path.as_os_str().to_string_lossy(),
        error
      )),
      Self::CannotReadFile { error } => f.write_fmt(format_args!("IO: Cannot read file: {}", error)),
      Self::CannotParseCSV { error } => f.write_fmt(format_args!("IO: Cannot parse CSV: {}", error)),
      Self::InvalidType { types } => f.write_fmt(format_args!("IO: Invalid tuple type: `{}`", types)),
      Self::ValueParseError { error } => std::fmt::Display::fmt(error, f),
      Self::CannotParseProbability { value } => f.write_fmt(format_args!("IO: Cannot parse probability `{}`", value)),
      Self::ArityMismatch { expected, found } => f.write_fmt(format_args!(
        "IO: Arity mismatch; expected {}, found {}",
        expected, found
      )),
      Self::CannotWriteRecord { error } => f.write_fmt(format_args!("IO: Cannot write record: {}", error)),
    }
  }
}
