use serde::ser::{Serialize, Serializer, SerializeStruct};

use crate::common::value_type::*;

use super::*;

#[derive(Debug, Clone, PartialEq, AstNode)]
#[doc(hidden)]
pub enum _Type {
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
  Named(String),
}

impl Serialize for _Type {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
      S: Serializer,
  {
    // 3 is the number of fields in the struct.
    let mut state = serializer.serialize_struct("_Type", 2)?;
    state.serialize_field("__kind__", "type")?;
    state.serialize_field("type", &format!("{}", self))?;
    state.end()
  }
}

impl std::fmt::Display for _Type {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::I8 => f.write_str("i8"),
      Self::I16 => f.write_str("i16"),
      Self::I32 => f.write_str("i32"),
      Self::I64 => f.write_str("i64"),
      Self::I128 => f.write_str("i128"),
      Self::ISize => f.write_str("isize"),
      Self::U8 => f.write_str("u8"),
      Self::U16 => f.write_str("u16"),
      Self::U32 => f.write_str("u32"),
      Self::U64 => f.write_str("u64"),
      Self::U128 => f.write_str("u128"),
      Self::USize => f.write_str("usize"),
      Self::F32 => f.write_str("f32"),
      Self::F64 => f.write_str("f64"),
      Self::Char => f.write_str("char"),
      Self::Bool => f.write_str("bool"),
      Self::Str => f.write_str("&str"),
      Self::String => f.write_str("String"),
      Self::Symbol => f.write_str("Symbol"),
      Self::DateTime => f.write_str("DateTime"),
      Self::Duration => f.write_str("Duration"),
      Self::Entity => f.write_str("Entity"),
      Self::Tensor => f.write_str("Tensor"),
      Self::Named(i) => f.write_str(i),
    }
  }
}

impl std::str::FromStr for Type {
  // There will be no error
  type Err = ();

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      "i8" => Ok(Self::i8()),
      "i16" => Ok(Self::i16()),
      "i32" => Ok(Self::i32()),
      "i64" => Ok(Self::i64()),
      "i128" => Ok(Self::i128()),
      "isize" => Ok(Self::isize()),
      "u8" => Ok(Self::u8()),
      "u16" => Ok(Self::u16()),
      "u32" => Ok(Self::u32()),
      "u64" => Ok(Self::u64()),
      "u128" => Ok(Self::u128()),
      "usize" => Ok(Self::usize()),
      "f32" => Ok(Self::f32()),
      "f64" => Ok(Self::f64()),
      "bool" => Ok(Self::bool()),
      "char" => Ok(Self::char()),
      "&str" => Ok(Self::str()),
      "String" => Ok(Self::string()),
      "Symbol" => Ok(Self::symbol()),
      "DateTime" => Ok(Self::datetime()),
      "Duration" => Ok(Self::duration()),
      "Entity" => Ok(Self::entity()),
      "Tensor" => Ok(Self::tensor()),
      s => Ok(Self::named(s.to_string())),
    }
  }
}

impl Type {
  /// Get the name of the type if the type node contains a custom named type
  pub fn get_name(&self) -> Option<&str> {
    if let Some(n) = self.as_named() {
      Some(n)
    } else {
      None
    }
  }

  /// Convert the type AST node to a value type
  ///
  /// Returns `Ok` if the node itself is a base type;
  /// `Err` if the node is a `Named` type and not normalized to base type
  pub fn to_value_type(&self) -> Result<ValueType, String> {
    match self.internal() {
      _Type::I8 => Ok(ValueType::I8),
      _Type::I16 => Ok(ValueType::I16),
      _Type::I32 => Ok(ValueType::I32),
      _Type::I64 => Ok(ValueType::I64),
      _Type::I128 => Ok(ValueType::I128),
      _Type::ISize => Ok(ValueType::ISize),
      _Type::U8 => Ok(ValueType::U8),
      _Type::U16 => Ok(ValueType::U16),
      _Type::U32 => Ok(ValueType::U32),
      _Type::U64 => Ok(ValueType::U64),
      _Type::U128 => Ok(ValueType::U128),
      _Type::USize => Ok(ValueType::USize),
      _Type::F32 => Ok(ValueType::F32),
      _Type::F64 => Ok(ValueType::F64),
      _Type::Char => Ok(ValueType::Char),
      _Type::Bool => Ok(ValueType::Bool),
      _Type::Str => Ok(ValueType::Str),
      _Type::String => Ok(ValueType::String),
      _Type::Symbol => Ok(ValueType::Symbol),
      _Type::DateTime => Ok(ValueType::DateTime),
      _Type::Duration => Ok(ValueType::Duration),
      _Type::Entity => Ok(ValueType::Entity),
      _Type::Tensor => Ok(ValueType::Tensor),
      _Type::Named(s) => Err(s.to_string()),
    }
  }
}

impl From<Identifier> for Type {
  fn from(value: Identifier) -> Self {
    let type_node = value
      .name()
      .parse()
      .expect("[Internal Error] Casting `Identifier` to `TypeNode` should not fail");
    Self::named_with_loc(type_node, value.location().clone())
  }
}
