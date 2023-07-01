use serde::*;

use crate::utils;

use super::tuple::*;
use super::value::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
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
  Symbol,
  DateTime,
  Duration,
  Entity,
  Tensor,
}

impl ValueType {
  pub fn type_of(prim: &Value) -> Self {
    use Value::*;
    match prim {
      I8(_) => Self::I8,
      I16(_) => Self::I16,
      I32(_) => Self::I32,
      I64(_) => Self::I64,
      I128(_) => Self::I128,
      ISize(_) => Self::ISize,
      U8(_) => Self::U8,
      U16(_) => Self::U16,
      U32(_) => Self::U32,
      U64(_) => Self::U64,
      U128(_) => Self::U128,
      USize(_) => Self::USize,
      F32(_) => Self::F32,
      F64(_) => Self::F64,
      Char(_) => Self::Char,
      Bool(_) => Self::Bool,
      Str(_) => Self::Str,
      String(_) => Self::String,
      Symbol(_) => Self::Symbol,
      SymbolString(_) => Self::Symbol,
      DateTime(_) => Self::DateTime,
      Duration(_) => Self::Duration,
      Entity(_) => Self::Entity,
      Tensor(_) => Self::Tensor,
      TensorValue(_) => Self::Tensor,
    }
  }

  pub fn zero(&self) -> Value {
    use Value as P;
    use ValueType::*;
    match self {
      I8 => P::I8(0),
      I16 => P::I16(0),
      I32 => P::I32(0),
      I64 => P::I64(0),
      I128 => P::I128(0),
      ISize => P::ISize(0),
      U8 => P::U8(0),
      U16 => P::U16(0),
      U32 => P::U32(0),
      U64 => P::U64(0),
      U128 => P::U128(0),
      USize => P::USize(0),
      F32 => P::F32(0.0),
      F64 => P::F64(0.0),
      _ => panic!("{:?} is not a numerical type", self),
    }
  }

  pub fn one(&self) -> Value {
    use Value as P;
    use ValueType::*;
    match self {
      I8 => P::I8(1),
      I16 => P::I16(1),
      I32 => P::I32(1),
      I64 => P::I64(1),
      I128 => P::I128(1),
      ISize => P::ISize(1),
      U8 => P::U8(1),
      U16 => P::U16(1),
      U32 => P::U32(1),
      U64 => P::U64(1),
      U128 => P::U128(1),
      USize => P::USize(1),
      F32 => P::F32(1.0),
      F64 => P::F64(1.0),
      _ => panic!("{:?} is not a numerical type", self),
    }
  }

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

  pub fn is_unsigned_integer(&self) -> bool {
    match self {
      Self::U8 | Self::U16 | Self::U32 | Self::U64 | Self::U128 | Self::USize => true,
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

  pub fn is_datetime(&self) -> bool {
    match self {
      Self::DateTime => true,
      _ => false,
    }
  }

  pub fn is_duration(&self) -> bool {
    match self {
      Self::Duration => true,
      _ => false,
    }
  }

  pub fn is_entity(&self) -> bool {
    match self {
      Self::Entity => true,
      _ => false,
    }
  }

  pub fn can_type_cast(&self, target: &Self) -> bool {
    if self.is_numeric() && target.is_numeric() {
      true
    } else if self.is_boolean() && target.is_boolean() {
      true
    } else if self.is_char() && (target.is_char() || target.is_string() || target.is_integer() || target.is_float()) {
      true
    } else if self.is_string() && target.is_numeric() {
      true
    } else {
      self.is_string() && target.is_string()
    }
  }

  pub fn parse(&self, s: &str) -> Result<Value, ValueParseError> {
    match self {
      // Signed
      Self::I8 => Ok(Value::I8(s.parse().map_err(|_| ValueParseError::new(s, self))?)),
      Self::I16 => Ok(Value::I16(s.parse().map_err(|_| ValueParseError::new(s, self))?)),
      Self::I32 => Ok(Value::I32(s.parse().map_err(|_| ValueParseError::new(s, self))?)),
      Self::I64 => Ok(Value::I64(s.parse().map_err(|_| ValueParseError::new(s, self))?)),
      Self::I128 => Ok(Value::I128(s.parse().map_err(|_| ValueParseError::new(s, self))?)),
      Self::ISize => Ok(Value::ISize(s.parse().map_err(|_| ValueParseError::new(s, self))?)),

      // Unsigned
      Self::U8 => Ok(Value::U8(s.parse().map_err(|_| ValueParseError::new(s, self))?)),
      Self::U16 => Ok(Value::U16(s.parse().map_err(|_| ValueParseError::new(s, self))?)),
      Self::U32 => Ok(Value::U32(s.parse().map_err(|_| ValueParseError::new(s, self))?)),
      Self::U64 => Ok(Value::U64(s.parse().map_err(|_| ValueParseError::new(s, self))?)),
      Self::U128 => Ok(Value::U128(s.parse().map_err(|_| ValueParseError::new(s, self))?)),
      Self::USize => Ok(Value::USize(s.parse().map_err(|_| ValueParseError::new(s, self))?)),

      // Floating point
      Self::F32 => Ok(Value::F32(s.parse().map_err(|_| ValueParseError::new(s, self))?)),
      Self::F64 => Ok(Value::F64(s.parse().map_err(|_| ValueParseError::new(s, self))?)),

      // Boolean
      Self::Bool => match s {
        "true" => Ok(Value::Bool(true)),
        "false" => Ok(Value::Bool(false)),
        _ => Err(ValueParseError::new(s, self)),
      },
      Self::Char => Ok(Value::Char(s.parse().map_err(|_| ValueParseError::new(s, self))?)),

      // String
      Self::Str => panic!("Cannot parse into a static string"),
      Self::String => Ok(Value::String(s.to_string())),
      Self::Symbol => panic!("Cannot parse into a symbol"),

      // DateTime and Duration
      Self::DateTime => Ok(Value::DateTime(
        utils::parse_date_time_string(s).ok_or_else(|| ValueParseError::new(s, self))?,
      )),
      Self::Duration => Ok(Value::Duration(
        utils::parse_duration_string(s).ok_or_else(|| ValueParseError::new(s, self))?,
      )),

      // Entity
      Self::Entity => panic!("Cannot parse into an entity"),

      // Tensor
      Self::Tensor => panic!("Cannot parse into tensor"),
    }
  }

  pub fn sum<'a, I: Iterator<Item = &'a Tuple>>(&self, i: I) -> Tuple {
    match self {
      Self::I8 => i.fold(0, |a, v| a + v.as_i8()).into(),
      Self::I16 => i.fold(0, |a, v| a + v.as_i16()).into(),
      Self::I32 => i.fold(0, |a, v| a + v.as_i32()).into(),
      Self::I64 => i.fold(0, |a, v| a + v.as_i64()).into(),
      Self::I128 => i.fold(0, |a, v| a + v.as_i128()).into(),
      Self::ISize => i.fold(0, |a, v| a + v.as_isize()).into(),

      // Unsigned
      Self::U8 => i.fold(0, |a, v| a + v.as_u8()).into(),
      Self::U16 => i.fold(0, |a, v| a + v.as_u16()).into(),
      Self::U32 => i.fold(0, |a, v| a + v.as_u32()).into(),
      Self::U64 => i.fold(0, |a, v| a + v.as_u64()).into(),
      Self::U128 => i.fold(0, |a, v| a + v.as_u128()).into(),
      Self::USize => i.fold(0, |a, v| a + v.as_usize()).into(),

      // Floating point
      Self::F32 => i.fold(0.0, |a, v| a + v.as_f32()).into(),
      Self::F64 => i.fold(0.0, |a, v| a + v.as_f64()).into(),

      // Others
      _ => panic!("Cannot perform sum on type `{}`", self),
    }
  }

  pub fn prod<'a, I: Iterator<Item = &'a Tuple>>(&self, i: I) -> Tuple {
    match self {
      Self::I8 => i.fold(1, |a, v| a * v.as_i8()).into(),
      Self::I16 => i.fold(1, |a, v| a * v.as_i16()).into(),
      Self::I32 => i.fold(1, |a, v| a * v.as_i32()).into(),
      Self::I64 => i.fold(1, |a, v| a * v.as_i64()).into(),
      Self::I128 => i.fold(1, |a, v| a * v.as_i128()).into(),
      Self::ISize => i.fold(1, |a, v| a * v.as_isize()).into(),

      // Unsigned
      Self::U8 => i.fold(1, |a, v| a * v.as_u8()).into(),
      Self::U16 => i.fold(1, |a, v| a * v.as_u16()).into(),
      Self::U32 => i.fold(1, |a, v| a * v.as_u32()).into(),
      Self::U64 => i.fold(1, |a, v| a * v.as_u64()).into(),
      Self::U128 => i.fold(1, |a, v| a * v.as_u128()).into(),
      Self::USize => i.fold(1, |a, v| a * v.as_usize()).into(),

      // Floating point
      Self::F32 => i.fold(1.0, |a, v| a * v.as_f32()).into(),
      Self::F64 => i.fold(1.0, |a, v| a * v.as_f64()).into(),

      // Others
      _ => panic!("Cannot perform sum on type `{}`", self),
    }
  }

  /// Get all integer types
  pub fn integers() -> &'static [ValueType] {
    &[
      ValueType::I8,
      ValueType::I16,
      ValueType::I32,
      ValueType::I64,
      ValueType::I128,
      ValueType::ISize,
      ValueType::U8,
      ValueType::U16,
      ValueType::U32,
      ValueType::U64,
      ValueType::U128,
      ValueType::USize,
    ]
  }

  /// Get all signed integer types
  pub fn signed_integers() -> &'static [ValueType] {
    &[
      ValueType::I8,
      ValueType::I16,
      ValueType::I32,
      ValueType::I64,
      ValueType::I128,
      ValueType::ISize,
    ]
  }

  /// Get all unsigned integer types
  pub fn unsigned_integers() -> &'static [ValueType] {
    &[
      ValueType::U8,
      ValueType::U16,
      ValueType::U32,
      ValueType::U64,
      ValueType::U128,
      ValueType::USize,
    ]
  }

  /// Get all floating point number types
  pub fn floats() -> &'static [ValueType] {
    &[ValueType::F32, ValueType::F64]
  }
}

#[derive(Clone, Debug)]
pub struct ValueParseError {
  pub source: String,
  pub ty: ValueType,
}

impl ValueParseError {
  pub fn new(s: &str, t: &ValueType) -> Self {
    Self {
      source: s.to_string(),
      ty: t.clone(),
    }
  }
}

impl std::fmt::Display for ValueParseError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("Cannot parse value `{}` as `{}`", self.source, self.ty))
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
      Symbol => f.write_str("Symbol"),
      DateTime => f.write_str("DateTime"),
      Duration => f.write_str("Duration"),
      Entity => f.write_str("Entity"),
      Tensor => f.write_str("Tensor"),
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

impl FromType<chrono::DateTime<chrono::Utc>> for ValueType {
  fn from_type() -> Self {
    Self::DateTime
  }
}
impl FromType<chrono::Duration> for ValueType {
  fn from_type() -> Self {
    Self::Duration
  }
}
