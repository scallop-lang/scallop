use std::path::PathBuf;

use super::super::source::*;
use crate::compiler::front::FrontCompileError;

#[derive(Clone, Debug)]
pub struct CycleImportError {
  pub path: PathBuf,
}

impl CycleImportError {
  pub fn report(&self, _: &Sources) {
    println!("File `{}` is already imported", self.path.to_str().unwrap());
  }
}

impl From<CycleImportError> for FrontCompileError {
  fn from(c: CycleImportError) -> Self {
    Self::CycleImportError(c)
  }
}

impl std::fmt::Display for CycleImportError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!(
      "File `{}` is already imported",
      self.path.to_str().unwrap()
    ))
  }
}
