use std::collections::*;

use crate::compiler::front;

#[derive(Clone, Debug, PartialEq)]
pub enum AttributeArgument {
  Float(f64),
  Integer(i64),
  Boolean(bool),
  String(String),
  List(Vec<AttributeArgument>),
}

impl AttributeArgument {
  pub fn to_front(&self) -> front::ast::AttributeValue {
    match self {
      Self::Float(f) => front::ast::Constant::float(f.clone()).into(),
      Self::Integer(i) => front::ast::Constant::integer(i.clone()).into(),
      Self::Boolean(b) => front::ast::Constant::boolean(b.clone()).into(),
      Self::String(s) => front::ast::Constant::string(s.clone()).into(),
      Self::List(l) => {
        let l = l.iter().map(AttributeArgument::to_front).collect();
        front::ast::AttributeValue::default(front::ast::AttributeValueNode::List(l))
      }
    }
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

impl<T> From<Vec<T>> for AttributeArgument
where
  T: Into<AttributeArgument>,
{
  fn from(v: Vec<T>) -> Self {
    Self::List(v.into_iter().map(|t| t.into()).collect())
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

  pub fn to_front(&self) -> front::ast::Attribute {
    front::ast::Attribute::default(front::ast::AttributeNode {
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

fn string_to_front_identifier(s: &str) -> front::ast::Identifier {
  front::ast::Identifier::default(front::ast::IdentifierNode { name: s.to_string() })
}
