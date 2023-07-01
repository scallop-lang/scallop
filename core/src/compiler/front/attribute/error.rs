use super::super::*;

#[derive(Clone, Debug)]
pub enum AttributeError {
  DuplicatedAttributeProcessor { name: String },
  ReservedAttribute { name: String },
  Custom { msg: String },
}

impl AttributeError {
  pub fn new_custom(msg: String) -> Self {
    Self::Custom { msg }
  }
}

impl std::fmt::Display for AttributeError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::DuplicatedAttributeProcessor { name } => {
        f.write_fmt(format_args!("Duplicated attribute processor `{}`", name))
      }
      Self::ReservedAttribute { name } => {
        f.write_fmt(format_args!("Attribute process `{}` is reserved in Scallop", name))
      }
      Self::Custom { msg } => f.write_str(msg),
    }
  }
}

impl FrontCompileErrorTrait for AttributeError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, _: &Sources) -> String {
    format!("{}", self)
  }
}
