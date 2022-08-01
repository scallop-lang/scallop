use super::super::*;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub enum SourceError {
  CannotOpenFile { file_name: PathBuf, std_io_error: String },
}

impl From<SourceError> for FrontCompileError {
  fn from(c: SourceError) -> Self {
    Self::SourceError(c)
  }
}

impl std::fmt::Display for SourceError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::CannotOpenFile {
        file_name,
        std_io_error,
      } => f.write_fmt(format_args!(
        "Cannot open file {}: {}\n",
        file_name.display(),
        std_io_error
      )),
    }
  }
}
