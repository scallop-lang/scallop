use crate::compiler;
use crate::runtime::error::*;

#[derive(Clone, Debug)]
pub enum IntegrateError {
  Compile(Vec<compiler::CompileError>),
  Runtime(RuntimeError),
}

impl IntegrateError {
  pub fn front(e: compiler::front::FrontCompileError) -> Self {
    Self::Compile(vec![compiler::CompileError::Front(e)])
  }
}

impl std::fmt::Display for IntegrateError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Compile(errs) => {
        for (i, e) in errs.iter().enumerate() {
          if i > 0 {
            f.write_str("\n")?;
          }
          std::fmt::Display::fmt(e, f)?;
        }
        Ok(())
      }
      Self::Runtime(e) => std::fmt::Display::fmt(e, f),
    }
  }
}
