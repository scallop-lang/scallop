use chrono::serde::ts_seconds;
use serde::Serialize;

use super::*;
use crate::common::input_tag::DynamicInputTag;
use crate::common::value::Value;
use crate::common::value_type::ValueType;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[doc(hidden)]
pub struct TagNode(pub DynamicInputTag);

/// A tag associated with a fact
pub type Tag = AstNode<TagNode>;

impl Tag {
  pub fn default_none() -> Self {
    Self::default(TagNode(DynamicInputTag::None))
  }

  pub fn input_tag(&self) -> &DynamicInputTag {
    &self.node.0
  }

  pub fn is_some(&self) -> bool {
    self.input_tag().is_some()
  }
}

#[derive(Clone, Debug, PartialEq, Hash, Serialize)]
pub struct ConstantCharNode {
  pub character: String,
}

pub type ConstantChar = AstNode<ConstantCharNode>;

impl ConstantChar {
  pub fn character(&self) -> char {
    // Unwrap is ok since during parsing
    self.node.character.chars().next().unwrap()
  }

  pub fn character_string(&self) -> &String {
    &self.node.character
  }
}

#[derive(Clone, Debug, PartialEq, Hash, Serialize)]
pub struct ConstantStringNode {
  pub string: String,
}

impl ConstantStringNode {
  pub fn new(string: String) -> Self {
    Self { string }
  }
}

pub type ConstantString = AstNode<ConstantStringNode>;

impl ConstantString {
  pub fn string(&self) -> &String {
    &self.node.string
  }

  pub fn string_mut(&mut self) -> &mut String {
    &mut self.node.string
  }
}

#[derive(Clone, Debug, PartialEq, Hash, Serialize)]
pub struct ConstantSymbolNode {
  pub symbol: String,
}

pub type ConstantSymbol = AstNode<ConstantSymbolNode>;

impl ConstantSymbol {
  pub fn symbol(&self) -> &String {
    &self.node.symbol
  }
}

#[derive(Clone, Debug, PartialEq, Hash, Serialize)]
pub struct ConstantDateTimeNode {
  #[serde(with = "ts_seconds")]
  pub datetime: chrono::DateTime<chrono::Utc>,
}

pub type ConstantDateTime = AstNode<Result<ConstantDateTimeNode, String>>;

impl ConstantDateTime {
  pub fn datetime(&self) -> Option<&chrono::DateTime<chrono::Utc>> {
    self.node.as_ref().ok().map(|n| &n.datetime)
  }
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct ConstantDurationNode {
  pub duration: chrono::Duration,
}

impl Serialize for ConstantDurationNode {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    use serde::ser::*;
    let mut state = serializer.serialize_struct("ConstantDurationNode", 1)?;
    state.serialize_field("duration", &self.duration.num_seconds())?;
    state.end()
  }
}

pub type ConstantDuration = AstNode<Result<ConstantDurationNode, String>>;

impl ConstantDuration {
  pub fn duration(&self) -> Option<&chrono::Duration> {
    self.node.as_ref().ok().map(|n| &n.duration)
  }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[doc(hidden)]
pub enum ConstantNode {
  Integer(i64),
  Float(f64),
  Char(ConstantChar),
  Boolean(bool),
  String(ConstantString),
  Symbol(ConstantSymbol),
  DateTime(ConstantDateTime),
  Duration(ConstantDuration),
  Entity(u64),
}

impl std::hash::Hash for ConstantNode {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    match self {
      Self::Integer(i) => i.hash(state),
      Self::Float(f) => i64::from_ne_bytes(f.to_ne_bytes()).hash(state),
      Self::Char(c) => c.hash(state),
      Self::Boolean(b) => b.hash(state),
      Self::String(s) => s.hash(state),
      Self::Symbol(s) => s.hash(state),
      Self::DateTime(d) => d.hash(state),
      Self::Duration(d) => d.hash(state),
      Self::Entity(u) => u.hash(state),
    }
  }
}

impl ConstantNode {
  pub fn kind(&self) -> &'static str {
    use ConstantNode::*;
    match self {
      Integer(_) => "integer",
      Float(_) => "float",
      String(_) => "string",
      Symbol(_) => "symbol",
      Char(_) => "char",
      Boolean(_) => "boolean",
      DateTime(_) => "datetime",
      Duration(_) => "duration",
      Entity(_) => "entity",
    }
  }
}

/// A constant, which could be an integer, floating point, character, boolean, or string.
pub type Constant = AstNode<ConstantNode>;

impl Constant {
  /// Create a new constant integer AST node
  pub fn integer(i: i64) -> Self {
    Self::default(ConstantNode::Integer(i))
  }

  pub fn float(f: f64) -> Self {
    Self::default(ConstantNode::Float(f))
  }

  pub fn boolean(b: bool) -> Self {
    Self::default(ConstantNode::Boolean(b))
  }

  pub fn string(s: String) -> Self {
    Self::default(ConstantNode::String(ConstantStringNode::new(s).into()))
  }

  pub fn can_unify(&self, ty: &ValueType) -> bool {
    use ConstantNode::*;
    match (&self.node, ty) {
      (Integer(_), ValueType::I8)
      | (Integer(_), ValueType::I16)
      | (Integer(_), ValueType::I32)
      | (Integer(_), ValueType::I64)
      | (Integer(_), ValueType::I128)
      | (Integer(_), ValueType::ISize)
      | (Integer(_), ValueType::U8)
      | (Integer(_), ValueType::U16)
      | (Integer(_), ValueType::U32)
      | (Integer(_), ValueType::U64)
      | (Integer(_), ValueType::U128)
      | (Integer(_), ValueType::USize)
      | (Integer(_), ValueType::F32)
      | (Integer(_), ValueType::F64)
      | (Float(_), ValueType::F32)
      | (Float(_), ValueType::F64)
      | (Char(_), ValueType::Char)
      | (Boolean(_), ValueType::Bool)
      | (String(_), ValueType::String)
      | (Symbol(_), ValueType::Symbol)
      | (DateTime(_), ValueType::DateTime)
      | (Duration(_), ValueType::Duration)
      | (Entity(_), ValueType::Entity) => true,
      _ => false,
    }
  }

  pub fn to_value(&self, ty: &ValueType) -> Value {
    use ConstantNode::*;
    match (&self.node, ty) {
      (Integer(i), ValueType::I8) => Value::I8(*i as i8),
      (Integer(i), ValueType::I16) => Value::I16(*i as i16),
      (Integer(i), ValueType::I32) => Value::I32(*i as i32),
      (Integer(i), ValueType::I64) => Value::I64(*i as i64),
      (Integer(i), ValueType::I128) => Value::I128(*i as i128),
      (Integer(i), ValueType::ISize) => Value::ISize(*i as isize),
      (Integer(i), ValueType::U8) => Value::U8(*i as u8),
      (Integer(i), ValueType::U16) => Value::U16(*i as u16),
      (Integer(i), ValueType::U32) => Value::U32(*i as u32),
      (Integer(i), ValueType::U64) => Value::U64(*i as u64),
      (Integer(i), ValueType::U128) => Value::U128(*i as u128),
      (Integer(i), ValueType::USize) => Value::USize(*i as usize),
      (Integer(i), ValueType::F32) => Value::F32(*i as f32),
      (Integer(i), ValueType::F64) => Value::F64(*i as f64),
      (Float(f), ValueType::F32) => Value::F32(*f as f32),
      (Float(f), ValueType::F64) => Value::F64(*f as f64),
      (Char(c), ValueType::Char) => Value::Char(c.character()),
      (Boolean(b), ValueType::Bool) => Value::Bool(*b),
      (String(_), ValueType::Str) => panic!("Cannot cast dynamic string into static string"),
      (String(s), ValueType::String) => Value::String(s.string().clone()),
      (Symbol(s), ValueType::Symbol) => Value::SymbolString(s.symbol().clone()),
      (DateTime(d), ValueType::DateTime) => {
        Value::DateTime(d.datetime().expect("Cannot have invalid datetime").clone())
      }
      (Duration(d), ValueType::Duration) => {
        Value::Duration(d.duration().expect("Cannot have invalid duration").clone())
      }
      (Entity(u), ValueType::Entity) => Value::Entity(*u),
      _ => panic!("Cannot convert front Constant `{:?}` to Type `{}`", self, ty),
    }
  }

  pub fn kind(&self) -> &'static str {
    self.node.kind()
  }

  pub fn as_bool(&self) -> Option<&bool> {
    match &self.node {
      ConstantNode::Boolean(b) => Some(b),
      _ => None,
    }
  }

  pub fn as_integer(&self) -> Option<&i64> {
    match &self.node {
      ConstantNode::Integer(i) => Some(i),
      _ => None,
    }
  }

  pub fn as_float(&self) -> Option<&f64> {
    match &self.node {
      ConstantNode::Float(f) => Some(f),
      _ => None,
    }
  }

  pub fn as_string(&self) -> Option<&String> {
    match &self.node {
      ConstantNode::String(s) => Some(s.string()),
      _ => None,
    }
  }

  pub fn as_char(&self) -> Option<&String> {
    match &self.node {
      ConstantNode::Char(c) => Some(c.character_string()),
      _ => None,
    }
  }

  pub fn as_datetime(&self) -> Option<&chrono::DateTime<chrono::Utc>> {
    match &self.node {
      ConstantNode::DateTime(d) => d.datetime(),
      _ => None,
    }
  }

  pub fn as_duration(&self) -> Option<&chrono::Duration> {
    match &self.node {
      ConstantNode::Duration(d) => d.duration(),
      _ => None,
    }
  }
}

/// A constant or a variable
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum ConstantOrVariable {
  Constant(Constant),
  Variable(Variable),
}

impl ConstantOrVariable {
  pub fn is_constant(&self) -> bool {
    match self {
      Self::Constant(_) => true,
      Self::Variable(_) => false,
    }
  }

  pub fn is_variable(&self) -> bool {
    match self {
      Self::Constant(_) => false,
      Self::Variable(_) => true,
    }
  }

  pub fn constant(&self) -> Option<&Constant> {
    match self {
      Self::Constant(c) => Some(c),
      Self::Variable(_) => None,
    }
  }

  pub fn variable(&self) -> Option<&Variable> {
    match self {
      Self::Constant(_) => None,
      Self::Variable(v) => Some(v),
    }
  }
}

#[derive(Clone, PartialEq, Serialize)]
#[doc(hidden)]
pub struct IdentifierNode {
  pub name: String,
}

impl IdentifierNode {
  pub fn new(name: String) -> Self {
    Self { name }
  }
}

/// An identifier, e.g. `predicate`
pub type Identifier = AstNode<IdentifierNode>;

impl Identifier {
  pub fn default_with_name(name: String) -> Self {
    IdentifierNode::new(name).into()
  }

  pub fn name(&self) -> &str {
    &self.node.name
  }

  pub fn map<F: FnOnce(&str) -> String>(&self, f: F) -> Self {
    Self {
      loc: self.loc.clone(),
      node: IdentifierNode {
        name: f(&self.node.name),
      },
    }
  }
}

impl std::fmt::Debug for IdentifierNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{:?}", &self.name))
  }
}
