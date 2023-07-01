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

  pub fn io(e: IOError) -> Self {
    Self::Runtime(RuntimeError::IO(e))
  }

  pub fn kind(&self) -> &'static str {
    match self {
      Self::Compile(_) => "Compile error occurred; aborted",
      Self::Runtime(_) => "Runtime error occurred; aborted",
    }
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
