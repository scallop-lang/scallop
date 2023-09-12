use serde::*;

use super::foreign_tensor::DynamicExternalTensor;

#[derive(Clone, Debug, PartialEq, PartialOrd, Serialize)]
pub enum DynamicInputTag {
  None,
  Exclusive(usize),
  Bool(bool),
  Natural(usize),
  Float(f64),
  ExclusiveFloat(f64, usize),
  Tensor(DynamicExternalTensor),
}

impl DynamicInputTag {
  pub fn is_some(&self) -> bool {
    match self {
      Self::None => false,
      _ => true,
    }
  }

  pub fn is_none(&self) -> bool {
    match self {
      Self::None => true,
      _ => false,
    }
  }

  pub fn with_exclusivity(&self, exclusion_id: usize) -> Self {
    match self {
      Self::None => Self::Exclusive(exclusion_id),
      Self::Float(f) => Self::ExclusiveFloat(f.clone(), exclusion_id),
      Self::ExclusiveFloat(f, _) => Self::ExclusiveFloat(f.clone(), exclusion_id),
      _ => self.clone(),
    }
  }
}

impl std::fmt::Display for DynamicInputTag {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::None => Ok(()),
      Self::Exclusive(i) => f.write_str(&format!("[ME({})]", i)),
      Self::Bool(b) => b.fmt(f),
      Self::Natural(n) => n.fmt(f),
      Self::Float(n) => n.fmt(f),
      Self::ExclusiveFloat(n, i) => f.write_str(&format!("{} [ME({})]", n, i)),
      Self::Tensor(t) => t.fmt(f),
    }
  }
}

impl Default for DynamicInputTag {
  fn default() -> Self {
    Self::None
  }
}

impl std::str::FromStr for DynamicInputTag {
  type Err = ParseInputTagError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    if s == "true" {
      Ok(Self::Bool(true))
    } else if s == "false" {
      Ok(Self::Bool(false))
    } else {
      let f = s.parse::<f64>().map_err(|_| ParseInputTagError {
        source_str: s.to_string(),
      })?;
      Ok(Self::Float(f))
    }
  }
}

pub struct ParseInputTagError {
  source_str: String,
}

impl std::fmt::Debug for ParseInputTagError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!(
      "[Parse Input Tag Error] Cannot parse `{}` into input tag",
      self.source_str
    ))
  }
}
