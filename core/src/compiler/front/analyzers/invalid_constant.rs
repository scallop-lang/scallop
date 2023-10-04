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

impl NodeVisitor<DateTimeLiteral> for InvalidConstantAnalyzer {
  fn visit(&mut self, datetime: &DateTimeLiteral) {
    match crate::utils::parse_date_time_string(datetime.datetime()) {
      Some(_) => {}
      None => {
        self.errors.push(InvalidConstantError::InvalidConstant {
          loc: datetime.location().clone(),
          message: format!("Invalid DateTime literal `{}`", datetime.datetime()),
        });
      }
    }
  }
}

impl NodeVisitor<DurationLiteral> for InvalidConstantAnalyzer {
  fn visit(&mut self, duration: &DurationLiteral) {
    match crate::utils::parse_duration_string(duration.duration()) {
      Some(_) => {}
      None => {
        self.errors.push(InvalidConstantError::InvalidConstant {
          loc: duration.location().clone(),
          message: format!("Invalid Duration literal `{}`", duration.duration()),
        });
      }
    }
  }
}

#[derive(Clone, Debug)]
pub enum InvalidConstantError {
  InvalidConstant { loc: NodeLocation, message: String },
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
