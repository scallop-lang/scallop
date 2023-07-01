use std::path::PathBuf;

use super::super::source::*;
use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub struct CycleImportError {
  pub path: PathBuf,
}

impl CycleImportError {
  pub fn report(&self, _: &Sources) {
    eprintln!("File `{}` is already imported", self.path.to_str().unwrap());
  }
}

impl FrontCompileErrorTrait for CycleImportError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, _: &Sources) -> String {
    format!("File `{}` is already imported", self.path.to_str().unwrap())
  }
}
