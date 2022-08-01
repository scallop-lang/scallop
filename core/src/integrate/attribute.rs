use std::collections::*;

use crate::compiler::front;

#[derive(Clone, Debug, PartialEq)]
pub enum AttributeArgument {
  Float(f64),
  Integer(i64),
  Boolean(bool),
  String(String),
}

impl AttributeArgument {
  pub fn to_front(&self) -> front::Constant {
    let c = match self {
      Self::Float(f) => front::ConstantNode::Float(f.clone()),
      Self::Integer(i) => front::ConstantNode::Integer(i.clone()),
      Self::Boolean(b) => front::ConstantNode::Boolean(b.clone()),
      Self::String(s) => front::ConstantNode::String(s.clone()),
    };
    front::Constant::default(c)
  }
}

impl From<String> for AttributeArgument {
  fn from(s: String) -> Self {
    Self::String(s)
  }
}

impl From<bool> for AttributeArgument {
  fn from(b: bool) -> Self {
    Self::Boolean(b)
  }
}

impl From<i64> for AttributeArgument {
  fn from(i: i64) -> Self {
    Self::Integer(i)
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Attribute {
  pub name: String,
  pub positional_arguments: Vec<AttributeArgument>,
  pub keyword_arguments: HashMap<String, AttributeArgument>,
}

impl Attribute {
  pub fn named(name: &str) -> Self {
    Self {
      name: name.to_string(),
      positional_arguments: vec![],
      keyword_arguments: HashMap::new(),
    }
  }

  pub fn to_front(&self) -> front::Attribute {
    front::Attribute::default(front::AttributeNode {
      name: string_to_front_identifier(&self.name),
      pos_args: self
        .positional_arguments
        .iter()
        .map(AttributeArgument::to_front)
        .collect(),
      kw_args: self
        .keyword_arguments
        .iter()
        .map(|(name, arg)| {
          let name = string_to_front_identifier(name);
          let arg = arg.to_front();
          (name, arg)
        })
        .collect(),
    })
  }
}

fn string_to_front_identifier(s: &str) -> front::Identifier {
  front::Identifier::default(front::IdentifierNode { name: s.to_string() })
}
