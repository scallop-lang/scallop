use super::super::ast::*;
use super::super::error::*;
use super::super::source::*;
use super::super::utils::*;
use super::super::visitor::*;

#[derive(Debug, Clone)]
pub struct FunctionAnalysis {
  pub errors: Vec<FunctionAnalysisError>,
}

impl FunctionAnalysis {
  pub fn new() -> Self {
    Self { errors: vec![] }
  }
}

impl NodeVisitor for FunctionAnalysis {
  fn visit_function(&mut self, function: &Function) {
    match &function.node {
      FunctionNode::Unknown(u) => self.errors.push(FunctionAnalysisError::UnknownFunction {
        function: u.clone(),
        loc: function.location().clone(),
      }),
      _ => {}
    }
  }
}

#[derive(Debug, Clone)]
pub enum FunctionAnalysisError {
  UnknownFunction { function: String, loc: Loc },
}

impl FrontCompileErrorTrait for FunctionAnalysisError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, src: &Sources) -> String {
    match self {
      Self::UnknownFunction { function, loc } => {
        format!("unknown function `{}`\n{}", function, loc.report(src))
      }
    }
  }
}

impl FrontCompileErrorClone for FunctionAnalysisError {
  fn clone_box(&self) -> Box<dyn FrontCompileErrorTrait> {
    Box::new(self.clone())
  }
}
