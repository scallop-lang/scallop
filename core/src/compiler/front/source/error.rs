use super::super::*;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub enum SourceError {
  CannotOpenFile { file_name: PathBuf, std_io_error: String },
}

impl FrontCompileErrorTrait for SourceError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, _: &Sources) -> String {
    match self {
      Self::CannotOpenFile {
        file_name,
        std_io_error,
      } => format!("Cannot open file {}: {}\n", file_name.display(), std_io_error),
    }
  }
}
