use std::path::PathBuf;

use crate::common::tuple_type::*;
use crate::common::value_type::*;

#[derive(Clone, Debug)]
pub enum IOError {
  CannotOpenFile { file_path: PathBuf, error: String },
  CannotReadFile { error: String },
  CannotParseCSV { error: String },
  CannotReadHeader { error: String },
  CannotFindField { field: String },
  IndexOutOfBounds { index: usize },
  InvalidType { types: TupleType },
  ExpectSymbolType { actual: ValueType },
  ExpectStringType { actual: ValueType },
  ValueParseError { error: ValueParseError },
  CannotParseProbability { value: String },
  ArityMismatch { expected: usize, found: usize },
  CannotWriteRecord { error: String },
  InvalidFileFormat {},
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
      Self::CannotReadHeader { error } => f.write_fmt(format_args!("IO: Cannot read CSV header: {}", error)),
      Self::CannotFindField { field } => f.write_fmt(format_args!("IO: Cannot find field `{}`", field)),
      Self::IndexOutOfBounds { index } => f.write_fmt(format_args!("IO: Index out of bounds: {}", index)),
      Self::InvalidType { types } => f.write_fmt(format_args!("IO: Invalid tuple type: `{}`", types)),
      Self::ExpectSymbolType { actual } => {
        f.write_fmt(format_args!("IO: Expect `Symbol` type for field; found `{}`", actual))
      }
      Self::ExpectStringType { actual } => {
        f.write_fmt(format_args!("IO: Expect `String` type for value; found `{}`", actual))
      }
      Self::ValueParseError { error } => std::fmt::Display::fmt(error, f),
      Self::CannotParseProbability { value } => f.write_fmt(format_args!("IO: Cannot parse probability `{}`", value)),
      Self::ArityMismatch { expected, found } => f.write_fmt(format_args!(
        "IO: Arity mismatch; expected {}, found {}",
        expected, found
      )),
      Self::CannotWriteRecord { error } => f.write_fmt(format_args!("IO: Cannot write record: {}", error)),
      Self::InvalidFileFormat {} => f.write_fmt(format_args!("IO: Invalid file format")),
    }
  }
}
