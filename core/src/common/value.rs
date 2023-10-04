use std::convert::*;

use chrono::{DateTime, Utc};
use chronoutil::RelativeDuration;

use super::duration::*;
use super::foreign_tensor::*;
use super::value_type::*;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Value {
  I8(i8),
  I16(i16),
  I32(i32),
  I64(i64),
  I128(i128),
  ISize(isize),
  U8(u8),
  U16(u16),
  U32(u32),
  U64(u64),
  U128(u128),
  USize(usize),
  F32(f32),
  F64(f64),
  Char(char),
  Bool(bool),
  Str(&'static str),
  String(String),
  Symbol(usize),
  SymbolString(String),
  DateTime(DateTime<Utc>),
  Duration(Duration),
  Entity(u64),
  EntityString(String),
  Tensor(DynamicExternalTensor),
  TensorValue(TensorValue),
}

impl Value {
  pub fn value_type(&self) -> ValueType {
    ValueType::type_of(self)
  }

  pub fn as_date_time(&self) -> DateTime<Utc> {
    match self {
      Self::DateTime(d) => d.clone(),
      _ => panic!("Not a DateTime"),
    }
  }

  pub fn as_duration(&self) -> Duration {
    match self {
      Self::Duration(d) => d.clone(),
      _ => panic!("Not a Duration"),
    }
  }

  pub fn as_usize(&self) -> usize {
    match self {
      Self::USize(u) => *u,
      v => panic!("Cannot cast value {} as usize", v),
    }
  }

  pub fn as_str(&self) -> &str {
    match self {
      Self::Str(s) => s,
      Self::String(s) => &s,
      v => panic!("Cannot get string from value {}", v),
    }
  }

  pub fn symbol_str(s: &str) -> Self {
    Self::SymbolString(s.to_string())
  }
}

impl Eq for Value {}

impl Ord for Value {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    match self.partial_cmp(other) {
      Some(o) => o,
      None => std::cmp::Ordering::Equal,
    }
  }
}

impl std::hash::Hash for Value {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    match self {
      Self::I8(i) => i.hash(state),
      Self::I16(i) => i.hash(state),
      Self::I32(i) => i.hash(state),
      Self::I64(i) => i.hash(state),
      Self::I128(i) => i.hash(state),
      Self::ISize(i) => i.hash(state),
      Self::U8(u) => u.hash(state),
      Self::U16(u) => u.hash(state),
      Self::U32(u) => u.hash(state),
      Self::U64(u) => u.hash(state),
      Self::U128(u) => u.hash(state),
      Self::USize(u) => u.hash(state),
      Self::F32(f) => i32::from_ne_bytes(f.to_ne_bytes()).hash(state),
      Self::F64(f) => i64::from_ne_bytes(f.to_ne_bytes()).hash(state),
      Self::Char(c) => c.hash(state),
      Self::Bool(b) => b.hash(state),
      Self::Str(s) => s.hash(state),
      Self::String(s) => s.hash(state),
      Self::Symbol(s) => s.hash(state),
      Self::SymbolString(_) => panic!("[Internal Error] Hash should not happen for symbol string"),
      Self::DateTime(d) => d.hash(state),
      Self::Duration(d) => d.hash(state),
      Self::Entity(e) => {
        "entity".hash(state);
        e.hash(state);
      }
      Self::EntityString(_) => panic!("[Internal Error] Hash should not happen for entity string"),
      Self::Tensor(_) => panic!("[Internal Error] Hash should not happen for tensor"),
      Self::TensorValue(v) => v.hash(state),
    }
  }
}

impl std::fmt::Display for Value {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::I8(i) => f.write_fmt(format_args!("{}", i)),
      Self::I16(i) => f.write_fmt(format_args!("{}", i)),
      Self::I32(i) => f.write_fmt(format_args!("{}", i)),
      Self::I64(i) => f.write_fmt(format_args!("{}", i)),
      Self::I128(i) => f.write_fmt(format_args!("{}", i)),
      Self::ISize(i) => f.write_fmt(format_args!("{}", i)),
      Self::U8(i) => f.write_fmt(format_args!("{}", i)),
      Self::U16(i) => f.write_fmt(format_args!("{}", i)),
      Self::U32(i) => f.write_fmt(format_args!("{}", i)),
      Self::U64(i) => f.write_fmt(format_args!("{}", i)),
      Self::U128(i) => f.write_fmt(format_args!("{}", i)),
      Self::USize(i) => f.write_fmt(format_args!("{}", i)),
      Self::F32(i) => f.write_fmt(format_args!("{}", i)),
      Self::F64(i) => f.write_fmt(format_args!("{}", i)),
      Self::Char(i) => f.write_fmt(format_args!("'{}'", i)),
      Self::Bool(i) => f.write_fmt(format_args!("{}", i)),
      Self::Str(i) => f.write_fmt(format_args!("{:?}", i)),
      Self::String(i) => f.write_fmt(format_args!("{:?}", i)),
      Self::Symbol(_) => panic!("[Internal Error] Cannot display symbol"),
      Self::SymbolString(s) => f.write_fmt(format_args!("s\"{}\"", s)),
      Self::DateTime(i) => f.write_fmt(format_args!("t\"{}\"", i)),
      Self::Duration(i) => f.write_fmt(format_args!("d\"{}\"", i)),
      Self::Entity(e) => f.write_fmt(format_args!("entity({e:#x})")),
      Self::EntityString(s) => f.write_fmt(format_args!("entity_string({s})")),
      Self::Tensor(t) => f.write_fmt(format_args!("{:?}", t)),
      Self::TensorValue(v) => f.write_fmt(format_args!("`{}`", v)),
    }
  }
}

impl From<i8> for Value {
  fn from(i: i8) -> Self {
    Self::I8(i)
  }
}

impl From<i16> for Value {
  fn from(i: i16) -> Self {
    Self::I16(i)
  }
}

impl From<i32> for Value {
  fn from(i: i32) -> Self {
    Self::I32(i)
  }
}

impl From<i64> for Value {
  fn from(i: i64) -> Self {
    Self::I64(i)
  }
}

impl From<i128> for Value {
  fn from(i: i128) -> Self {
    Self::I128(i)
  }
}

impl From<isize> for Value {
  fn from(i: isize) -> Self {
    Self::ISize(i)
  }
}

impl From<u8> for Value {
  fn from(u: u8) -> Self {
    Self::U8(u)
  }
}

impl From<u16> for Value {
  fn from(u: u16) -> Self {
    Self::U16(u)
  }
}

impl From<u32> for Value {
  fn from(u: u32) -> Self {
    Self::U32(u)
  }
}

impl From<u64> for Value {
  fn from(u: u64) -> Self {
    Self::U64(u)
  }
}

impl From<u128> for Value {
  fn from(u: u128) -> Self {
    Self::U128(u)
  }
}

impl From<usize> for Value {
  fn from(u: usize) -> Self {
    Self::USize(u)
  }
}

impl From<f32> for Value {
  fn from(f: f32) -> Self {
    Self::F32(f)
  }
}

impl From<f64> for Value {
  fn from(f: f64) -> Self {
    Self::F64(f)
  }
}

impl From<char> for Value {
  fn from(c: char) -> Self {
    Self::Char(c)
  }
}

impl From<bool> for Value {
  fn from(b: bool) -> Self {
    Self::Bool(b)
  }
}

impl From<&'static str> for Value {
  fn from(s: &'static str) -> Self {
    Self::Str(s)
  }
}

impl From<String> for Value {
  fn from(s: String) -> Self {
    Self::String(s)
  }
}

impl From<DateTime<Utc>> for Value {
  fn from(dt: DateTime<Utc>) -> Self {
    Self::DateTime(dt)
  }
}

impl From<RelativeDuration> for Value {
  fn from(d: RelativeDuration) -> Self {
    Self::Duration(d.into())
  }
}

macro_rules! impl_try_into {
  ($into_ty:ty, $variant:ident) => {
    impl TryInto<$into_ty> for Value {
      type Error = ValueConversionError;

      fn try_into(self) -> Result<$into_ty, Self::Error> {
        match self {
          Self::$variant(i) => Ok(i),
          _ => Err(ValueConversionError),
        }
      }
    }
  };
}

#[derive(Clone, Debug, Default)]
pub struct ValueConversionError;

impl_try_into!(i8, I8);
impl_try_into!(i16, I16);
impl_try_into!(i32, I32);
impl_try_into!(i64, I64);
impl_try_into!(i128, I128);
impl_try_into!(isize, ISize);
impl_try_into!(u8, U8);
impl_try_into!(u16, U16);
impl_try_into!(u32, U32);
impl_try_into!(u64, U64);
impl_try_into!(u128, U128);
impl_try_into!(usize, USize);
impl_try_into!(f32, F32);
impl_try_into!(f64, F64);
impl_try_into!(bool, Bool);
impl_try_into!(char, Char);
impl_try_into!(String, String);
