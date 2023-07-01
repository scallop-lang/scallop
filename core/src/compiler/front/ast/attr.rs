use std::iter::FromIterator;

use serde::*;

use super::*;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[doc(hidden)]
pub enum AttributeValueNode {
  Constant(Constant),
  List(Vec<AttributeValue>),
  Tuple(Vec<AttributeValue>),
}

/// The value of an attribute; it could be a list or a constant
pub type AttributeValue = AstNode<AttributeValueNode>;

impl AttributeValue {
  pub fn constant(c: Constant) -> Self {
    Self::default(AttributeValueNode::Constant(c))
  }

  pub fn is_constant(&self) -> bool {
    match &self.node {
      AttributeValueNode::Constant(_) => true,
      _ => false,
    }
  }

  pub fn is_list(&self) -> bool {
    match &self.node {
      AttributeValueNode::List(_) => true,
      _ => false,
    }
  }

  pub fn is_tuple(&self) -> bool {
    match &self.node {
      AttributeValueNode::Tuple(_) => true,
      _ => false,
    }
  }

  pub fn as_constant(&self) -> Option<&Constant> {
    match &self.node {
      AttributeValueNode::Constant(c) => Some(c),
      _ => None,
    }
  }

  pub fn as_bool(&self) -> Option<&bool> {
    self.as_constant().and_then(Constant::as_bool)
  }

  pub fn as_integer(&self) -> Option<&i64> {
    self.as_constant().and_then(Constant::as_integer)
  }

  pub fn as_string(&self) -> Option<&String> {
    self.as_constant().and_then(Constant::as_string)
  }

  pub fn as_list(&self) -> Option<&Vec<AttributeValue>> {
    match &self.node {
      AttributeValueNode::List(l) => Some(l),
      _ => None,
    }
  }
}

impl From<Constant> for AttributeValue {
  fn from(c: Constant) -> Self {
    Self::new(c.location().clone_without_id(), AttributeValueNode::Constant(c))
  }
}

impl FromIterator<AttributeValue> for AttributeValue {
  fn from_iter<T: IntoIterator<Item = AttributeValue>>(iter: T) -> Self {
    Self::default(AttributeValueNode::List(iter.into_iter().collect()))
  }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[doc(hidden)]
pub struct AttributeNode {
  pub name: Identifier,
  pub pos_args: Vec<AttributeValue>,
  pub kw_args: Vec<(Identifier, AttributeValue)>,
}

/// An attribute of the form `@attr(args...)`
pub type Attribute = AstNode<AttributeNode>;

impl Attribute {
  pub fn default_with_name(n: String) -> Self {
    AttributeNode {
      name: Identifier::default_with_name(n),
      pos_args: Vec::new(),
      kw_args: Vec::new(),
    }
    .into()
  }

  pub fn name(&self) -> &String {
    &self.node.name.node.name
  }

  pub fn num_pos_args(&self) -> usize {
    self.node.pos_args.len()
  }

  pub fn pos_arg(&self, i: usize) -> Option<&AttributeValue> {
    self.node.pos_args.get(i)
  }

  pub fn pos_arg_to_bool(&self, i: usize) -> Option<&bool> {
    self
      .node
      .pos_args
      .get(i)
      .and_then(AttributeValue::as_constant)
      .and_then(Constant::as_bool)
  }

  pub fn pos_arg_to_integer(&self, i: usize) -> Option<&i64> {
    self
      .node
      .pos_args
      .get(i)
      .and_then(AttributeValue::as_constant)
      .and_then(Constant::as_integer)
  }

  pub fn pos_arg_to_string(&self, i: usize) -> Option<&String> {
    self
      .node
      .pos_args
      .get(i)
      .and_then(AttributeValue::as_constant)
      .and_then(Constant::as_string)
  }

  pub fn pos_arg_to_list(&self, i: usize) -> Option<&Vec<AttributeValue>> {
    self.node.pos_args.get(i).and_then(AttributeValue::as_list)
  }

  pub fn iter_pos_args(&self) -> impl Iterator<Item = &AttributeValue> {
    self.node.pos_args.iter()
  }

  pub fn num_kw_args(&self) -> usize {
    self.node.kw_args.len()
  }

  pub fn kw_arg(&self, kw: &str) -> Option<&AttributeValue> {
    for (name, arg) in &self.node.kw_args {
      if name.name() == kw {
        return Some(arg);
      }
    }
    None
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
      if attr.name() == name {
        return Some(attr);
      }
    }
    None
  }
}
