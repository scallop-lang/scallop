#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum DynamicInputTag {
  None,
  Exclusive(usize),
  Bool(bool),
  Float(f64),
  ExclusiveFloat(f64, usize),
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
      Self::Float(n) => n.fmt(f),
      Self::ExclusiveFloat(n, i) => f.write_str(&format!("{} [ME({})]", n, i)),
    }
  }
}

impl Default for DynamicInputTag {
  fn default() -> Self {
    Self::None
  }
}

pub trait StaticInputTag: Sized {
  fn from_dynamic_input_tag(_: &DynamicInputTag) -> Option<Self>;
}

impl<T> StaticInputTag for T {
  default fn from_dynamic_input_tag(_: &DynamicInputTag) -> Option<Self> {
    None
  }
}

impl StaticInputTag for bool {
  fn from_dynamic_input_tag(t: &DynamicInputTag) -> Option<Self> {
    match t {
      DynamicInputTag::Bool(b) => Some(b.clone()),
      _ => None,
    }
  }
}

impl StaticInputTag for f64 {
  fn from_dynamic_input_tag(t: &DynamicInputTag) -> Option<Self> {
    match t {
      DynamicInputTag::Float(f) => Some(f.clone()),
      DynamicInputTag::ExclusiveFloat(f, _) => Some(f.clone()),
      _ => None,
    }
  }
}

impl StaticInputTag for (f64, Option<usize>) {
  fn from_dynamic_input_tag(t: &DynamicInputTag) -> Option<Self> {
    match t {
      DynamicInputTag::Exclusive(i) => Some((1.0, Some(i.clone()))),
      DynamicInputTag::Float(f) => Some((f.clone(), None)),
      DynamicInputTag::ExclusiveFloat(f, u) => Some((f.clone(), Some(u.clone()))),
      _ => None,
    }
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
