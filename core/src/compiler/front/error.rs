use colored::*;

use super::analyzers::errors::*;
use super::parser::*;
use super::*;

#[derive(Clone, Debug)]
pub enum FrontCompileError {
  SourceError(SourceError),
  ParserError(ParserError),
  CycleImportError(CycleImportError),
  InvalidWildcardError(InvalidWildcardError),
  TypeInferenceError(TypeInferenceError),
  InputFilesError(InputFilesError),
  OutputFilesError(OutputFilesError),
  BoundnessAnalysisError(BoundnessAnalysisError),
  AggregationAnalysisError(AggregationAnalysisError),
  DemandAttributeError(DemandAttributeError),
}

impl FrontCompileError {
  pub fn report(&self, src: &Sources) {
    print!("{} ", "[Error]".red());
    match self {
      Self::SourceError(e) => println!("{}", e),
      Self::ParserError(e) => e.report(src),
      Self::CycleImportError(e) => e.report(src),
      Self::InputFilesError(e) => e.report(src),
      Self::OutputFilesError(e) => e.report(src),
      Self::InvalidWildcardError(e) => e.report(src),
      Self::TypeInferenceError(e) => e.report(src),
      Self::BoundnessAnalysisError(e) => e.report(src),
      Self::AggregationAnalysisError(e) => e.report(src),
      Self::DemandAttributeError(e) => e.report(src),
    }
  }
}

impl std::fmt::Display for FrontCompileError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::SourceError(e) => std::fmt::Display::fmt(e, f),
      Self::ParserError(e) => std::fmt::Display::fmt(e, f),
      Self::CycleImportError(e) => std::fmt::Display::fmt(e, f),
      Self::InputFilesError(e) => std::fmt::Display::fmt(e, f),
      Self::OutputFilesError(e) => std::fmt::Display::fmt(e, f),
      Self::InvalidWildcardError(e) => f.write_fmt(format_args!("{:?}", e)),
      Self::TypeInferenceError(e) => f.write_fmt(format_args!("{:?}", e)),
      Self::BoundnessAnalysisError(e) => f.write_fmt(format_args!("{:?}", e)),
      Self::AggregationAnalysisError(e) => f.write_fmt(format_args!("{:?}", e)),
      Self::DemandAttributeError(e) => f.write_fmt(format_args!("{:?}", e)),
    }
  }
}

#[derive(Clone, Debug)]
pub struct FrontErrorReportingContext {
  pub sources: Sources,
  pub errors: Vec<FrontCompileError>,
}

impl FrontErrorReportingContext {
  pub fn new() -> Self {
    Self {
      sources: Sources::new(),
      errors: Vec::new(),
    }
  }

  pub fn set_sources(&mut self, sources: &Sources) {
    self.sources = sources.clone();
  }

  pub fn add<T>(&mut self, error: T)
  where
    FrontCompileError: From<T>,
  {
    self.errors.push(FrontCompileError::from(error));
  }

  pub fn extend<T>(&mut self, new_errors: &mut Vec<T>)
  where
    FrontCompileError: From<T>,
  {
    self.errors.extend(new_errors.drain(..).map(FrontCompileError::from))
  }

  pub fn has_error(&self) -> bool {
    !self.errors.is_empty()
  }

  pub fn report_errors(&self) {
    for error in &self.errors {
      error.report(&self.sources);
      println!();
    }
  }

  pub fn clear_errors(&mut self) {
    self.errors.clear();
  }
}
