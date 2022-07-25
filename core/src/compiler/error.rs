use super::back::BackCompileError;
use super::front::FrontCompileError;

#[derive(Clone, Debug)]
pub enum CompileError {
  Front(FrontCompileError),
  Back(BackCompileError),
}

impl std::fmt::Display for CompileError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Front(e) => std::fmt::Display::fmt(e, f),
      Self::Back(e) => std::fmt::Display::fmt(e, f),
    }
  }
}

pub type CompileErrors = Vec<CompileError>;
