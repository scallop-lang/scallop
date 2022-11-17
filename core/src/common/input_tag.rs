#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum ExtInputTag<T: Clone + PartialEq + PartialOrd> {
  None,
  Bool(bool),
  Float(f64),
  TaggedFloat(f64, T),
}

pub type InputTag = ExtInputTag<()>;

impl<T: Clone + PartialEq + PartialOrd> ExtInputTag<T> {
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
}

impl<T: Clone + PartialEq + PartialOrd> std::fmt::Display for ExtInputTag<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::None => Ok(()),
      Self::Bool(b) => b.fmt(f),
      Self::Float(n) => n.fmt(f),
      Self::TaggedFloat(n, _) => n.fmt(f),
    }
  }
}

impl<T: Clone + PartialEq + PartialOrd> Default for ExtInputTag<T> {
  fn default() -> Self {
    Self::None
  }
}

pub trait FromInputTag: Sized {
  fn from_input_tag<T: Clone + PartialEq + PartialOrd>(t: &ExtInputTag<T>) -> Option<Self>;
}

impl<T> FromInputTag for T {
  default fn from_input_tag<TP: Clone + PartialEq + PartialOrd>(_: &ExtInputTag<TP>) -> Option<T> {
    None
  }
}

impl FromInputTag for bool {
  fn from_input_tag<TP: Clone + PartialEq + PartialOrd>(t: &ExtInputTag<TP>) -> Option<bool> {
    match t {
      ExtInputTag::Bool(b) => Some(b.clone()),
      _ => None,
    }
  }
}

impl FromInputTag for f64 {
  fn from_input_tag<TP: Clone + PartialEq + PartialOrd>(t: &ExtInputTag<TP>) -> Option<f64> {
    match t {
      ExtInputTag::Float(f) => Some(f.clone()),
      _ => None,
    }
  }
}

impl<TP: Clone + PartialEq + PartialOrd> std::str::FromStr for ExtInputTag<TP> {
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
