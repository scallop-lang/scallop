#[derive(Clone, Debug, PartialEq)]
pub enum AggregateOp {
  Count,
  Sum,
  Prod,
  Min,
  Max,
  Exists,
  Forall,
  Unique,
}

impl std::fmt::Display for AggregateOp {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Count => f.write_str("count"),
      Self::Sum => f.write_str("sum"),
      Self::Prod => f.write_str("prod"),
      Self::Min => f.write_str("min"),
      Self::Max => f.write_str("max"),
      Self::Exists => f.write_str("exists"),
      Self::Forall => f.write_str("forall"),
      Self::Unique => f.write_str("unique"),
    }
  }
}
