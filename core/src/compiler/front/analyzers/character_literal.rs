use super::super::ast::*;
use super::super::error::*;
use super::super::source::*;
use super::super::utils::*;
use super::super::visitor::*;

#[derive(Debug, Clone)]
pub struct CharacterLiteralAnalysis {
  pub errors: Vec<CharacterLiteralAnalysisError>,
}

impl CharacterLiteralAnalysis {
  pub fn new() -> Self {
    Self { errors: vec![] }
  }
}

impl NodeVisitor for CharacterLiteralAnalysis {
  fn visit_constant_char(&mut self, c: &ConstantChar) {
    let s = c.character_string();
    let loc = c.location().clone();
    if s.len() == 1 {
      // OK
    } else if s.len() == 0 {
      self.errors.push(CharacterLiteralAnalysisError::EmptyCharacter { loc })
    } else {
      self
        .errors
        .push(CharacterLiteralAnalysisError::InvalidCharacter { loc })
    }
  }
}

#[derive(Debug, Clone)]
pub enum CharacterLiteralAnalysisError {
  EmptyCharacter { loc: Loc },
  InvalidCharacter { loc: Loc },
}

impl FrontCompileErrorTrait for CharacterLiteralAnalysisError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, src: &Sources) -> String {
    match self {
      Self::EmptyCharacter { loc } => {
        format!("empty character at\n{}", loc.report(src))
      }
      Self::InvalidCharacter { loc } => {
        format!("invalid character at\n{}", loc.report(src))
      }
    }
  }
}
