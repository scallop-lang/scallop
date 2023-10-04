//! # Aggregate Operations

use super::value_type::*;

/// The aggregate operators for low level representation
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AggregateOp {
  Count { discrete: bool },
  Sum { has_arg: bool, ty: ValueType },
  Prod { has_arg: bool, ty: ValueType },
  Min,
  Argmin,
  Max,
  Argmax,
  Exists,
  TopK(usize),
  CategoricalK(usize),
}

impl std::fmt::Display for AggregateOp {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Count { discrete } => {
        if *discrete {
          f.write_str("discrete_count")
        } else {
          f.write_str("count")
        }
      }
      Self::Sum { has_arg, ty } => {
        if *has_arg {
          f.write_fmt(format_args!("sum_wa<{}>", ty))
        } else {
          f.write_fmt(format_args!("sum<{}>", ty))
        }
      }
      Self::Prod { has_arg, ty } => {
        if *has_arg {
          f.write_fmt(format_args!("prod_wa<{}>", ty))
        } else {
          f.write_fmt(format_args!("prod<{}>", ty))
        }
      }
      Self::Min => f.write_str("min"),
      Self::Max => f.write_str("max"),
      Self::Argmin => f.write_str("argmin"),
      Self::Argmax => f.write_str("argmax"),
      Self::Exists => f.write_str("exists"),
      Self::TopK(k) => f.write_fmt(format_args!("top<{}>", k)),
      Self::CategoricalK(k) => f.write_fmt(format_args!("categorical<{}>", k)),
    }
  }
}

impl AggregateOp {
  pub fn count() -> Self {
    Self::Count { discrete: false }
  }

  pub fn discrete_count() -> Self {
    Self::Count { discrete: true }
  }

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

  pub fn categorical_k(k: usize) -> Self {
    Self::CategoricalK(k)
  }

  pub fn is_min_max(&self) -> bool {
    match self {
      Self::Min | Self::Max | Self::Argmin | Self::Argmax => true,
      _ => false,
    }
  }

  pub fn is_sum_prod(&self) -> bool {
    match self {
      Self::Sum { .. } | Self::Prod { .. } => true,
      _ => false,
    }
  }
}
