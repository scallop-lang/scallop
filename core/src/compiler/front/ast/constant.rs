use crate::common::input_tag::DynamicInputTag;
use crate::common::value::Value;
use crate::common::value_type::ValueType;

use super::*;

/// A tag associated with a fact
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _Tag {
  pub tag: DynamicInputTag,
}

impl Tag {
  pub fn none() -> Self {
    Self::new(DynamicInputTag::None)
  }

  pub fn is_some(&self) -> bool {
    self.tag().is_some()
  }
}

#[derive(Clone, Debug, PartialEq, Hash, Serialize, AstNode)]
#[doc(hidden)]
pub struct _IntLiteral {
  pub int: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _FloatLiteral {
  pub float: f64,
}

impl std::hash::Hash for _FloatLiteral {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    i64::from_ne_bytes(self.float.to_ne_bytes()).hash(state)
  }
}

#[derive(Clone, Debug, PartialEq, Hash, Serialize, AstNode)]
#[doc(hidden)]
pub struct _BoolLiteral {
  pub value: bool,
}

#[derive(Clone, Debug, PartialEq, Hash, Serialize, AstNode)]
#[doc(hidden)]
pub struct _CharLiteral {
  pub character: String,
}

impl CharLiteral {
  pub fn parse_char(&self) -> char {
    // Unwrap is ok since during parsing we already know there is only one character
    self.character().chars().next().unwrap()
  }
}

#[derive(Clone, Debug, PartialEq, Hash, Serialize, AstNode)]
pub struct _StringLiteral {
  pub string: String,
}

#[derive(Clone, Debug, PartialEq, Hash, Serialize, AstNode)]
pub struct _SymbolLiteral {
  pub symbol: String,
}

#[derive(Clone, Debug, PartialEq, Hash, Serialize, AstNode)]
pub struct _DateTimeLiteral {
  pub datetime: String,
}

#[derive(Clone, Debug, PartialEq, Hash, Serialize, AstNode)]
pub struct _DurationLiteral {
  pub duration: String,
}

#[derive(Clone, Debug, PartialEq, Hash, Serialize, AstNode)]
pub struct _EntityLiteral {
  pub symbol: u64,
}

/// A constant, which could be an integer, floating point, character, boolean, or string.
#[derive(Clone, Debug, PartialEq, Hash, Serialize, AstNode)]
#[doc(hidden)]
pub enum Constant {
  Integer(IntLiteral),
  Float(FloatLiteral),
  Boolean(BoolLiteral),
  Char(CharLiteral),
  String(StringLiteral),
  Symbol(SymbolLiteral),
  DateTime(DateTimeLiteral),
  Duration(DurationLiteral),
  Entity(EntityLiteral),
}

impl Constant {
  pub fn can_unify(&self, ty: &ValueType) -> bool {
    match (self, ty) {
      (Self::Integer(_), ValueType::I8)
      | (Self::Integer(_), ValueType::I16)
      | (Self::Integer(_), ValueType::I32)
      | (Self::Integer(_), ValueType::I64)
      | (Self::Integer(_), ValueType::I128)
      | (Self::Integer(_), ValueType::ISize)
      | (Self::Integer(_), ValueType::U8)
      | (Self::Integer(_), ValueType::U16)
      | (Self::Integer(_), ValueType::U32)
      | (Self::Integer(_), ValueType::U64)
      | (Self::Integer(_), ValueType::U128)
      | (Self::Integer(_), ValueType::USize)
      | (Self::Integer(_), ValueType::F32)
      | (Self::Integer(_), ValueType::F64)
      | (Self::Float(_), ValueType::F32)
      | (Self::Float(_), ValueType::F64)
      | (Self::Char(_), ValueType::Char)
      | (Self::Boolean(_), ValueType::Bool)
      | (Self::String(_), ValueType::String)
      | (Self::Symbol(_), ValueType::Symbol)
      | (Self::DateTime(_), ValueType::DateTime)
      | (Self::Duration(_), ValueType::Duration)
      | (Self::Entity(_), ValueType::Entity) => true,
      _ => false,
    }
  }

  pub fn to_value(&self, ty: &ValueType) -> Value {
    match (self, ty) {
      (Self::Integer(i), ValueType::I8) => Value::I8(*i.int() as i8),
      (Self::Integer(i), ValueType::I16) => Value::I16(*i.int() as i16),
      (Self::Integer(i), ValueType::I32) => Value::I32(*i.int() as i32),
      (Self::Integer(i), ValueType::I64) => Value::I64(*i.int() as i64),
      (Self::Integer(i), ValueType::I128) => Value::I128(*i.int() as i128),
      (Self::Integer(i), ValueType::ISize) => Value::ISize(*i.int() as isize),
      (Self::Integer(i), ValueType::U8) => Value::U8(*i.int() as u8),
      (Self::Integer(i), ValueType::U16) => Value::U16(*i.int() as u16),
      (Self::Integer(i), ValueType::U32) => Value::U32(*i.int() as u32),
      (Self::Integer(i), ValueType::U64) => Value::U64(*i.int() as u64),
      (Self::Integer(i), ValueType::U128) => Value::U128(*i.int() as u128),
      (Self::Integer(i), ValueType::USize) => Value::USize(*i.int() as usize),
      (Self::Integer(i), ValueType::F32) => Value::F32(*i.int() as f32),
      (Self::Integer(i), ValueType::F64) => Value::F64(*i.int() as f64),
      (Self::Float(f), ValueType::F32) => Value::F32(f.float().clone() as f32),
      (Self::Float(f), ValueType::F64) => Value::F64(f.float().clone()),
      (Self::Char(c), ValueType::Char) => Value::Char(c.parse_char()),
      (Self::Boolean(b), ValueType::Bool) => Value::Bool(b.value().clone()),
      (Self::String(_), ValueType::Str) => panic!("Cannot cast dynamic string into static string"),
      (Self::String(s), ValueType::String) => Value::String(s.string().clone()),
      (Self::Symbol(s), ValueType::Symbol) => Value::SymbolString(s.symbol().clone()),
      (Self::DateTime(d), ValueType::DateTime) => {
        Value::DateTime(crate::utils::parse_date_time_string(d.datetime()).expect("Cannot have invalid datetime"))
      }
      (Self::Duration(d), ValueType::Duration) => {
        Value::Duration(crate::utils::parse_duration_string(d.duration()).expect("Cannot have invalid duration"))
      }
      (Self::Entity(u), ValueType::Entity) => Value::Entity(u.symbol().clone()),
      _ => panic!("Cannot convert front Constant `{:?}` to Type `{}`", self, ty),
    }
  }

  pub fn kind(&self) -> &'static str {
    match self {
      Self::Integer(_) => "integer",
      Self::Float(_) => "float",
      Self::String(_) => "string",
      Self::Symbol(_) => "symbol",
      Self::Char(_) => "char",
      Self::Boolean(_) => "boolean",
      Self::DateTime(_) => "datetime",
      Self::Duration(_) => "duration",
      Self::Entity(_) => "entity",
    }
  }
}

/// A constant or a variable
#[derive(Clone, Debug, PartialEq, Hash, Serialize, AstNode)]
pub enum ConstantOrVariable {
  Constant(Constant),
  Variable(Variable),
}

/// An identifier, e.g. `predicate`
#[derive(Clone, PartialEq, Hash, Serialize, AstNode)]
#[doc(hidden)]
pub struct _Identifier {
  pub name: String,
}

impl Identifier {
  pub fn map<F: FnOnce(&str) -> String>(&self, f: F) -> Self {
    Self {
      _loc: self.location().clone(),
      _node: _Identifier::new(f(self.name())),
    }
  }
}

impl std::fmt::Debug for _Identifier {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{:?}", &self.name))
  }
}
