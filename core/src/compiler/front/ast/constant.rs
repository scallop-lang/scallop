// use std::rc::Rc;

use super::*;
use crate::common::{input_tag::InputTag, value::Value, value_type::ValueType};

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct TagNode(pub InputTag);

/// A tag associated with a fact
pub type Tag = AstNode<TagNode>;

impl Tag {
  pub fn default_none() -> Self {
    Self::default(TagNode(InputTag::None))
  }

  pub fn input_tag(&self) -> &InputTag {
    &self.node.0
  }

  pub fn is_some(&self) -> bool {
    self.input_tag().is_some()
  }
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub enum ConstantNode {
  Integer(i64),
  Float(f64),
  Char(String),
  Boolean(bool),
  String(String),
}

/// A constant, which could be an integer, floating point, character, boolean, or string.
pub type Constant = AstNode<ConstantNode>;

impl Constant {
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
      (Char(c), ValueType::Char) => Value::Char(c.chars().next().unwrap()),
      (Boolean(b), ValueType::Bool) => Value::Bool(*b),
      (String(_), ValueType::Str) => panic!("Cannot cast dynamic string into static string"),
      (String(s), ValueType::String) => Value::String(s.clone()),
      // (String(s), ValueType::RcString) => Value::RcString(Rc::new(s.clone())),
      _ => panic!("Cannot convert front Constant `{:?}` to Type `{}`", self, ty),
    }
  }

  pub fn kind(&self) -> &'static str {
    use ConstantNode::*;
    match &self.node {
      Integer(_) => "integer",
      Float(_) => "float",
      String(_) => "string",
      Char(_) => "char",
      Boolean(_) => "boolean",
    }
  }
}

/// A constant or a variable
#[derive(Clone, Debug, PartialEq)]
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

#[derive(Clone, PartialEq)]
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
  pub fn name(&self) -> &str {
    &self.node.name
  }
}

impl std::fmt::Debug for IdentifierNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{:?}", &self.name))
  }
}
