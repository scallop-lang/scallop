use crate::compiler::front::ast;

#[derive(Clone, Debug, PartialEq)]
pub enum AttributeValue {
  Float(f64),
  Integer(i64),
  Boolean(bool),
  String(String),
  List(Vec<AttributeValue>),
  Tuple(Vec<AttributeValue>),
}

impl AttributeValue {
  pub fn string(s: String) -> Self {
    Self::String(s)
  }

  pub fn to_front(&self) -> ast::AttributeValue {
    match self {
      Self::Float(f) => ast::AttributeValue::float(f.clone()),
      Self::Integer(i) => ast::AttributeValue::integer(i.clone()),
      Self::Boolean(b) => ast::AttributeValue::boolean(b.clone()),
      Self::String(s) => ast::AttributeValue::string(s.clone()),
      Self::List(l) => {
        let l = l.iter().map(AttributeValue::to_front).collect();
        ast::AttributeValue::list(ast::AttributeValueList::new(l))
      }
      Self::Tuple(l) => {
        let l = l.iter().map(AttributeValue::to_front).collect();
        ast::AttributeValue::tuple(ast::AttributeValueTuple::new(l))
      }
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub enum AttributeArgument {
  Positional(AttributeValue),
  Keyword(String, AttributeValue),
}

impl AttributeArgument {
  pub fn string(s: String) -> Self {
    Self::Positional(AttributeValue::String(s))
  }

  pub fn named_string(n: &str, s: String) -> Self {
    Self::Keyword(n.to_string(), AttributeValue::String(s))
  }

  pub fn named_bool(n: &str, b: bool) -> Self {
    Self::Keyword(n.to_string(), AttributeValue::Boolean(b))
  }

  pub fn named_list(n: &str, l: Vec<AttributeValue>) -> Self {
    Self::Keyword(n.to_string(), AttributeValue::List(l))
  }

  pub fn to_front(&self) -> ast::AttributeArg {
    match self {
      Self::Positional(v) => ast::AttributeArg::Pos(v.to_front()),
      Self::Keyword(n, v) => {
        let id = ast::Identifier::new(n.clone());
        let val = v.to_front();
        ast::AttributeArg::Kw(ast::AttributeKwArg::new(id, val))
      }
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Attribute {
  pub name: String,
  pub args: Vec<AttributeArgument>,
}

impl Attribute {
  pub fn named(name: &str) -> Self {
    Self {
      name: name.to_string(),
      args: vec![],
    }
  }

  pub fn iter_args(&self) -> impl Iterator<Item = &AttributeArgument> {
    self.args.iter()
  }

  pub fn to_front(&self) -> ast::Attribute {
    ast::Attribute::new(
      ast::Identifier::new(self.name.clone()),
      self.iter_args().map(AttributeArgument::to_front).collect(),
    )
  }
}
