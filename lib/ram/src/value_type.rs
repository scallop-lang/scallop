#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValueType {
  I8,
  I16,
  I32,
  I64,
  I128,
  ISize,
  U8,
  U16,
  U32,
  U64,
  U128,
  USize,
  F32,
  F64,
  Char,
  Bool,
  Str,
  String,
}

impl ValueType {
  pub fn is_numeric(&self) -> bool {
    self.is_integer() || self.is_float()
  }

  pub fn is_integer(&self) -> bool {
    match self {
      Self::I8
      | Self::I16
      | Self::I32
      | Self::I64
      | Self::I128
      | Self::ISize
      | Self::U8
      | Self::U16
      | Self::U32
      | Self::U64
      | Self::U128
      | Self::USize => true,
      _ => false,
    }
  }

  pub fn is_signed_integer(&self) -> bool {
    match self {
      Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::I128 | Self::ISize => true,
      _ => false,
    }
  }

  pub fn is_float(&self) -> bool {
    match self {
      Self::F32 | Self::F64 => true,
      _ => false,
    }
  }

  pub fn is_boolean(&self) -> bool {
    match self {
      Self::Bool => true,
      _ => false,
    }
  }

  pub fn is_char(&self) -> bool {
    match self {
      Self::Char => true,
      _ => false,
    }
  }

  pub fn is_string(&self) -> bool {
    match self {
      Self::Str | Self::String /* | Self::RcString */ => true,
      _ => false,
    }
  }
}

impl std::fmt::Display for ValueType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use ValueType::*;
    match self {
      I8 => f.write_str("i8"),
      I16 => f.write_str("i16"),
      I32 => f.write_str("i32"),
      I64 => f.write_str("i64"),
      I128 => f.write_str("i128"),
      ISize => f.write_str("isize"),
      U8 => f.write_str("u8"),
      U16 => f.write_str("u16"),
      U32 => f.write_str("u32"),
      U64 => f.write_str("u64"),
      U128 => f.write_str("u128"),
      USize => f.write_str("usize"),
      F32 => f.write_str("f32"),
      F64 => f.write_str("f64"),
      Char => f.write_str("char"),
      Bool => f.write_str("bool"),
      Str => f.write_str("&str"),
      String => f.write_str("String"),
    }
  }
}

pub trait FromType<T> {
  fn from_type() -> Self;
}

impl FromType<i8> for ValueType {
  fn from_type() -> Self {
    Self::I8
  }
}

impl FromType<i16> for ValueType {
  fn from_type() -> Self {
    Self::I16
  }
}

impl FromType<i32> for ValueType {
  fn from_type() -> Self {
    Self::I32
  }
}

impl FromType<i64> for ValueType {
  fn from_type() -> Self {
    Self::I64
  }
}

impl FromType<i128> for ValueType {
  fn from_type() -> Self {
    Self::I128
  }
}

impl FromType<isize> for ValueType {
  fn from_type() -> Self {
    Self::ISize
  }
}

impl FromType<u8> for ValueType {
  fn from_type() -> Self {
    Self::U8
  }
}

impl FromType<u16> for ValueType {
  fn from_type() -> Self {
    Self::U16
  }
}

impl FromType<u32> for ValueType {
  fn from_type() -> Self {
    Self::U32
  }
}

impl FromType<u64> for ValueType {
  fn from_type() -> Self {
    Self::U64
  }
}

impl FromType<u128> for ValueType {
  fn from_type() -> Self {
    Self::U128
  }
}

impl FromType<usize> for ValueType {
  fn from_type() -> Self {
    Self::USize
  }
}

impl FromType<f32> for ValueType {
  fn from_type() -> Self {
    Self::F32
  }
}

impl FromType<f64> for ValueType {
  fn from_type() -> Self {
    Self::F64
  }
}

impl FromType<char> for ValueType {
  fn from_type() -> Self {
    Self::Char
  }
}

impl FromType<bool> for ValueType {
  fn from_type() -> Self {
    Self::Bool
  }
}

impl FromType<&'static str> for ValueType {
  fn from_type() -> Self {
    Self::Str
  }
}

impl FromType<String> for ValueType {
  fn from_type() -> Self {
    Self::String
  }
}
