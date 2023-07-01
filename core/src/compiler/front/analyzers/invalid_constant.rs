use super::super::*;

#[derive(Clone, Debug)]
pub struct InvalidConstantAnalyzer {
  pub errors: Vec<InvalidConstantError>,
}

impl InvalidConstantAnalyzer {
  pub fn new() -> Self {
    Self { errors: Vec::new() }
  }
}

impl NodeVisitor for InvalidConstantAnalyzer {
  fn visit_constant_datetime(&mut self, constant: &ConstantDateTime) {
    match &constant.node {
      Err(message) => {
        self.errors.push(InvalidConstantError::InvalidConstant {
          loc: constant.location().clone(),
          message: message.clone(),
        });
      }
      _ => {}
    }
  }

  fn visit_constant_duration(&mut self, constant: &ConstantDuration) {
    match &constant.node {
      Err(message) => {
        self.errors.push(InvalidConstantError::InvalidConstant {
          loc: constant.location().clone(),
          message: message.clone(),
        });
      }
      _ => {}
    }
  }
}

#[derive(Clone, Debug)]
pub enum InvalidConstantError {
  InvalidConstant { loc: AstNodeLocation, message: String },
}

impl FrontCompileErrorTrait for InvalidConstantError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, src: &Sources) -> String {
    match self {
      Self::InvalidConstant { loc, message } => {
        format!("Invalid constant: {}\n{}", message, loc.report(src))
      }
    }
  }
}
