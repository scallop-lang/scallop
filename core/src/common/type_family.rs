/// The type families enum
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeFamily {
  /// All possible types
  Any,

  /// All integers and floating point numbers
  Number,

  /// All integers (signed and unsigned)
  Integer,

  /// All signed integer of different sizes (8, 16, 32, ..., 128)
  SignedInteger,

  /// All unsigned integer of different sizes (8, 16, 32, ..., 128)
  UnsignedInteger,

  /// Floating point numbers of different sizes (32, 64)
  Float,

  /// All strings
  String,

  /// No type
  Bottom,
}

impl std::fmt::Display for TypeFamily {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Any => f.write_str("Any"),
      Self::Number => f.write_str("Number"),
      Self::Integer => f.write_str("Integer"),
      Self::SignedInteger => f.write_str("SignedInteger"),
      Self::UnsignedInteger => f.write_str("UnsignedInteger"),
      Self::Float => f.write_str("Float"),
      Self::String => f.write_str("String"),
      Self::Bottom => f.write_str("Bottom"),
    }
  }
}

impl std::cmp::PartialOrd for TypeFamily {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    match (self, other) {
      // First start from any
      (Self::Any, Self::Any) => Some(std::cmp::Ordering::Equal),
      (Self::Any, _) => Some(std::cmp::Ordering::Greater),

      // Then start from number
      (Self::Number, Self::Any) => Some(std::cmp::Ordering::Less),
      (Self::Number, Self::Number) => Some(std::cmp::Ordering::Equal),
      (Self::Number, Self::Integer)
      | (Self::Number, Self::SignedInteger)
      | (Self::Number, Self::UnsignedInteger)
      | (Self::Number, Self::Float)
      | (Self::Number, Self::Bottom) => Some(std::cmp::Ordering::Greater),

      // Then start from integer
      (Self::Integer, Self::Number) | (Self::Integer, Self::Any) => Some(std::cmp::Ordering::Less),
      (Self::Integer, Self::Integer) => Some(std::cmp::Ordering::Equal),
      (Self::Integer, Self::SignedInteger) | (Self::Integer, Self::UnsignedInteger) | (Self::Integer, Self::Bottom) => {
        Some(std::cmp::Ordering::Greater)
      }

      // Then signed integer
      (Self::SignedInteger, Self::Integer) | (Self::SignedInteger, Self::Number) | (Self::SignedInteger, Self::Any) => {
        Some(std::cmp::Ordering::Less)
      }
      (Self::SignedInteger, Self::SignedInteger) => Some(std::cmp::Ordering::Equal),
      (Self::SignedInteger, Self::Bottom) => Some(std::cmp::Ordering::Greater),

      // Then unsigned integer
      (Self::UnsignedInteger, Self::Integer)
      | (Self::UnsignedInteger, Self::Number)
      | (Self::UnsignedInteger, Self::Any) => Some(std::cmp::Ordering::Less),
      (Self::UnsignedInteger, Self::UnsignedInteger) => Some(std::cmp::Ordering::Equal),
      (Self::UnsignedInteger, Self::Bottom) => Some(std::cmp::Ordering::Greater),

      // Then float
      (Self::Float, Self::Number) | (Self::Float, Self::Any) => Some(std::cmp::Ordering::Less),
      (Self::Float, Self::Float) => Some(std::cmp::Ordering::Equal),
      (Self::Float, Self::Bottom) => Some(std::cmp::Ordering::Greater),

      // Then string
      (Self::String, Self::Any) => Some(std::cmp::Ordering::Less),
      (Self::String, Self::String) => Some(std::cmp::Ordering::Equal),
      (Self::String, Self::Bottom) => Some(std::cmp::Ordering::Greater),

      // Lastly bottom
      (Self::Bottom, Self::Bottom) => Some(std::cmp::Ordering::Equal),
      (Self::Bottom, _) => Some(std::cmp::Ordering::Less),

      // There is no ordering for any other things
      _ => None,
    }
  }
}
