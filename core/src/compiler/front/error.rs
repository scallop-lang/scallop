use colored::*;

use super::*;

pub enum FrontCompileErrorType {
  Warning,
  Error,
}

impl FrontCompileErrorType {
  pub fn marker(&self) -> String {
    match self {
      Self::Warning => format!("{}", "[Warning]".yellow()),
      Self::Error => format!("{}", "[Error]".red()),
    }
  }

  pub fn is_warning(&self) -> bool {
    match self {
      Self::Warning => true,
      _ => false,
    }
  }

  pub fn is_error(&self) -> bool {
    match self {
      Self::Error => true,
      _ => false,
    }
  }
}

pub trait FrontCompileErrorTrait: FrontCompileErrorClone + std::fmt::Debug {
  /// Get the error type of this error (warning/error)
  fn error_type(&self) -> FrontCompileErrorType;

  /// Report the error showing source into string
  fn report(&self, src: &Sources) -> String;
}

pub trait FrontCompileErrorClone {
  fn clone_box(&self) -> Box<dyn FrontCompileErrorTrait>;
}

impl Clone for Box<dyn FrontCompileErrorTrait> {
  fn clone(&self) -> Box<dyn FrontCompileErrorTrait> {
    self.clone_box()
  }
}

#[derive(Clone, Debug)]
pub struct FrontCompileError {
  pub sources: Sources,
  pub errors: Vec<Box<dyn FrontCompileErrorTrait>>,
}

impl std::fmt::Display for FrontCompileError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for error in &self.errors {
      f.write_fmt(format_args!(
        "{} {}\n",
        error.error_type().marker(),
        error.report(&self.sources)
      ))?;
    }
    Ok(())
  }
}

impl FrontCompileError {
  pub fn new() -> Self {
    Self {
      sources: Sources::new(),
      errors: Vec::new(),
    }
  }

  pub fn singleton<E: FrontCompileErrorTrait + 'static>(e: E) -> Self {
    Self {
      sources: Sources::new(),
      errors: vec![Box::new(e)],
    }
  }

  pub fn set_sources(&mut self, sources: &Sources) {
    self.sources = sources.clone();
  }

  pub fn add<E: FrontCompileErrorTrait + 'static>(&mut self, error: E) {
    self.errors.push(Box::new(error));
  }

  pub fn extend<E: FrontCompileErrorTrait + 'static>(&mut self, new_errors: &mut Vec<E>) {
    for e in new_errors.drain(..) {
      self.add(e)
    }
  }

  pub fn has_error(&self) -> bool {
    self.errors.iter().any(|e| e.error_type().is_error())
  }

  pub fn has_warning(&self) -> bool {
    self.errors.iter().any(|e| e.error_type().is_warning())
  }

  pub fn report_errors(&self) {
    println!("{}", self)
  }

  pub fn report_warnings(&self) {
    for error in &self.errors {
      if error.error_type().is_warning() {
        println!("{} {}\n", error.error_type().marker(), error.report(&self.sources))
      }
    }
  }

  pub fn clear_errors(&mut self) {
    self.errors.clear();
  }
}
