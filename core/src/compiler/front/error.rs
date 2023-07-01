use colored::*;
use dyn_clone::DynClone;

use crate::common::value_type::ValueParseError;

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

#[derive(Debug)]
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

impl Clone for FrontCompileError {
  fn clone(&self) -> Self {
    Self {
      sources: self.sources.clone(),
      errors: self.errors.iter().map(|e| dyn_clone::clone_box(&**e)).collect(),
    }
  }
}

impl FrontCompileError {
  pub fn new() -> Self {
    Self {
      sources: Sources::new(),
      errors: Vec::new(),
    }
  }

  pub fn with_sources(sources: &Sources) -> Self {
    Self {
      sources: sources.clone(),
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

  pub fn add_source<S: Source>(&mut self, source: S) -> usize {
    self.sources.add(source)
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
    eprintln!("{}", self)
  }

  pub fn report_warnings(&self) {
    for error in &self.errors {
      if error.error_type().is_warning() {
        eprintln!("{} {}\n", error.error_type().marker(), error.report(&self.sources))
      }
    }
  }

  pub fn clear_errors(&mut self) {
    self.errors.clear();
  }
}

pub trait FrontCompileErrorTrait: DynClone + std::fmt::Debug {
  /// Get the error type of this error (warning/error)
  fn error_type(&self) -> FrontCompileErrorType;

  /// Report the error showing source into string
  fn report(&self, src: &Sources) -> String;
}

impl FrontCompileErrorTrait for ValueParseError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, _: &Sources) -> String {
    format!("cannot parse value `{}` into type `{}`", self.source, self.ty)
  }
}
