use super::*;

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct AttributeNode {
  pub name: Identifier,
  pub pos_args: Vec<Constant>,
  pub kw_args: Vec<(Identifier, Constant)>,
}

/// An attribute of the form `@attr(args...)`
pub type Attribute = AstNode<AttributeNode>;

impl Attribute {
  pub fn name(&self) -> &String {
    &self.node.name.node.name
  }

  pub fn num_pos_args(&self) -> usize {
    self.node.pos_args.len()
  }

  pub fn pos_arg(&self, i: usize) -> Option<&Constant> {
    self.node.pos_args.get(i)
  }

  pub fn iter_pos_args(&self) -> impl Iterator<Item = &Constant> {
    self.node.pos_args.iter()
  }

  pub fn num_kw_args(&self) -> usize {
    self.node.kw_args.len()
  }

  pub fn kw_arg(&self, kw: &str) -> Option<&Constant> {
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
