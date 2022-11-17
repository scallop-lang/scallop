use super::value_type::*;

/// The aggregate operators for low level representation
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AggregateOp {
  Count,
  Sum(ValueType),
  Prod(ValueType),
  Min,
  Argmin,
  Max,
  Argmax,
  Exists,
  TopK(usize),
}

impl std::fmt::Display for AggregateOp {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Count => f.write_str("count"),
      Self::Sum(t) => f.write_fmt(format_args!("sum<{}>", t)),
      Self::Prod(t) => f.write_fmt(format_args!("prod<{}>", t)),
      Self::Min => f.write_str("min"),
      Self::Max => f.write_str("max"),
      Self::Argmin => f.write_str("argmin"),
      Self::Argmax => f.write_str("argmax"),
      Self::Exists => f.write_str("exists"),
      Self::TopK(k) => f.write_fmt(format_args!("top<{}>", k)),
    }
  }
}

impl AggregateOp {
  pub fn min(has_arg: bool) -> Self {
    if has_arg {
      Self::Argmin
    } else {
      Self::Min
    }
  }

  pub fn max(has_arg: bool) -> Self {
    if has_arg {
      Self::Argmax
    } else {
      Self::Max
    }
  }

  pub fn top_k(k: usize) -> Self {
    Self::TopK(k)
  }
}
