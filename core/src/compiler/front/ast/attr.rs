use std::iter::FromIterator;

use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _AttributeValueList {
  pub values: Vec<AttributeValue>,
}

impl AttributeValueList {
  pub fn iter(&self) -> impl Iterator<Item = &AttributeValue> {
    self.values().iter()
  }

  pub fn is_empty(&self) -> bool {
    self.values().is_empty()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _AttributeValueTuple {
  pub values: Vec<AttributeValue>,
}

impl AttributeValueTuple {
  pub fn iter(&self) -> impl Iterator<Item = &AttributeValue> {
    self.values().iter()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
pub enum AttributeValue {
  Constant(Constant),
  List(AttributeValueList),
  Tuple(AttributeValueTuple),
}

impl AttributeValue {
  pub fn integer(i: i64) -> Self {
    Self::constant(Constant::integer(IntLiteral::new(i)))
  }

  pub fn float(f: f64) -> Self {
    Self::constant(Constant::float(FloatLiteral::new(f)))
  }

  pub fn boolean(b: bool) -> Self {
    Self::constant(Constant::boolean(BoolLiteral::new(b)))
  }

  pub fn character(c: char) -> Self {
    Self::constant(Constant::char(CharLiteral::new(c.into())))
  }

  pub fn string(s: String) -> Self {
    Self::constant(Constant::string(StringLiteral::new(s)))
  }

  pub fn as_integer(&self) -> Option<i64> {
    self.as_constant().and_then(|c| c.as_integer()).map(|b| b.int().clone())
  }

  pub fn as_float(&self) -> Option<f64> {
    self.as_constant().and_then(|c| c.as_float()).map(|b| b.float().clone())
  }

  pub fn as_boolean(&self) -> Option<bool> {
    self.as_constant().and_then(|c| c.as_boolean()).map(|b| b.value().clone())
  }

  pub fn as_string(&self) -> Option<String> {
    self.as_constant().and_then(|c| c.as_string()).map(|b| b.string().clone())
  }
}

impl FromIterator<AttributeValue> for AttributeValue {
  fn from_iter<T: IntoIterator<Item = AttributeValue>>(iter: T) -> Self {
    AttributeValue::list(AttributeValueList::new(iter.into_iter().collect()))
  }
}

impl Into<AttributeArg> for AttributeValue {
  fn into(self) -> AttributeArg {
    AttributeArg::Pos(self)
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _AttributeKwArg {
  pub name: Identifier,
  pub value: AttributeValue,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub enum AttributeArg {
  Pos(AttributeValue),
  Kw(AttributeKwArg),
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _Attribute {
  pub name: Identifier,
  pub args: Vec<AttributeArg>,
}

impl Attribute {
  pub fn attr_name(&self) -> &String {
    self.name().name()
  }

  pub fn num_pos_args(&self) -> usize {
    self.iter_args().filter(|a| AttributeArg::is_pos(a)).fold(0, |acc, _| acc + 1)
  }

  pub fn iter_pos_args(&self) -> impl Iterator<Item = &AttributeValue> {
    self.iter_args().filter_map(|a| AttributeArg::as_pos(a))
  }

  pub fn pos_arg(&self, i: usize) -> Option<&AttributeValue> {
    self.iter_pos_args().nth(i)
  }

  pub fn pos_arg_to_bool(&self, i: usize) -> Option<&bool> {
    self
      .pos_arg(i)
      .and_then(AttributeValue::as_constant)
      .and_then(Constant::as_boolean)
      .map(BoolLiteral::value)
  }

  pub fn pos_arg_to_integer(&self, i: usize) -> Option<&i64> {
    self
      .pos_arg(i)
      .and_then(AttributeValue::as_constant)
      .and_then(Constant::as_integer)
      .map(IntLiteral::int)
  }

  pub fn pos_arg_to_string(&self, i: usize) -> Option<&String> {
    self
      .pos_arg(i)
      .and_then(AttributeValue::as_constant)
      .and_then(Constant::as_string)
      .map(StringLiteral::string)
  }

  pub fn pos_arg_to_list(&self, i: usize) -> Option<&Vec<AttributeValue>> {
    self
      .pos_arg(i)
      .and_then(AttributeValue::as_list)
      .map(AttributeValueList::values)
  }

  pub fn num_kw_args(&self) -> usize {
    self.iter_args().filter(|a| AttributeArg::is_kw(a)).fold(0, |acc, _| acc + 1)
  }

  pub fn iter_kw_args(&self) -> impl Iterator<Item = &AttributeKwArg> {
    self.iter_args().filter_map(|a| AttributeArg::as_kw(a))
  }

  pub fn kw_arg(&self, kw: &str) -> Option<&AttributeValue> {
    self
      .iter_kw_args()
      .find(|kw_arg| kw_arg.name().name() == kw)
      .map(|kw_arg| kw_arg.value())
  }
}

/// A list of attributes
pub type Attributes = Vec<Attribute>;

pub trait AttributesTrait {
  fn find(&self, name: &str) -> Option<&Attribute>;
}

impl AttributesTrait for Attributes {
  fn find(&self, name: &str) -> Option<&Attribute> {
    for attr in self {
      if attr.attr_name() == name {
        return Some(attr);
      }
    }
    None
  }
}
