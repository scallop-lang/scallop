use super::*;
use crate::common::value_type::*;

#[derive(Debug, Clone, PartialEq)]
#[doc(hidden)]
pub enum TypeNode {
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
  // RcString,
  DateTime,
  Duration,
  Named(Identifier),
}

impl std::fmt::Display for TypeNode {
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
      // Self::RcString => f.write_str("Rc<String>"),
      Self::DateTime => f.write_str("DateTime"),
      Self::Duration => f.write_str("Duration"),
      Self::Named(i) => f.write_str(&i.node.name),
    }
  }
}

pub type Type = AstNode<TypeNode>;

impl Type {
  /// Create a new `i8` type AST node
  pub fn i8() -> Self {
    Self::default(TypeNode::I8)
  }

  /// Create a new `usize` type AST node
  pub fn usize() -> Self {
    Self::default(TypeNode::USize)
  }

  /// Convert the type AST node to a value type
  ///
  /// Returns `Ok` if the node itself is a base type;
  /// `Err` if the node is a `Named` type and not normalized to base type
  pub fn to_value_type(&self) -> Result<ValueType, String> {
    match &self.node {
      TypeNode::I8 => Ok(ValueType::I8),
      TypeNode::I16 => Ok(ValueType::I16),
      TypeNode::I32 => Ok(ValueType::I32),
      TypeNode::I64 => Ok(ValueType::I64),
      TypeNode::I128 => Ok(ValueType::I128),
      TypeNode::ISize => Ok(ValueType::ISize),
      TypeNode::U8 => Ok(ValueType::U8),
      TypeNode::U16 => Ok(ValueType::U16),
      TypeNode::U32 => Ok(ValueType::U32),
      TypeNode::U64 => Ok(ValueType::U64),
      TypeNode::U128 => Ok(ValueType::U128),
      TypeNode::USize => Ok(ValueType::USize),
      TypeNode::F32 => Ok(ValueType::F32),
      TypeNode::F64 => Ok(ValueType::F64),
      TypeNode::Char => Ok(ValueType::Char),
      TypeNode::Bool => Ok(ValueType::Bool),
      TypeNode::Str => Ok(ValueType::Str),
      TypeNode::String => Ok(ValueType::String),
      // TypeNode::RcString => Ok(ValueType::RcString),
      TypeNode::DateTime => Ok(ValueType::DateTime),
      TypeNode::Duration => Ok(ValueType::Duration),
      TypeNode::Named(s) => Err(s.name().to_string()),
    }
  }
}
