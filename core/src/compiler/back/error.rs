use super::*;

#[derive(Debug, Clone)]
pub enum BackCompileError {
  SCCError(SCCError),
  DemandTransformError(optimizations::DemandTransformError),
}

impl std::fmt::Display for BackCompileError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::SCCError(e) => std::fmt::Display::fmt(e, f),
      Self::DemandTransformError(e) => std::fmt::Display::fmt(e, f),
    }
  }
}
