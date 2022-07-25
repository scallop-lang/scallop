use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct InputDeclNode {
  pub attrs: Attributes,
  pub name: Identifier,
  pub types: Vec<ArgTypeBinding>,
}

pub type InputDecl = AstNode<InputDeclNode>;

impl InputDecl {
  pub fn attributes(&self) -> &Attributes {
    &self.node.attrs
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    &mut self.node.attrs
  }

  pub fn predicate(&self) -> &str {
    self.node.name.name()
  }

  pub fn arity(&self) -> usize {
    self.node.types.len()
  }

  pub fn iter_attributes(&self) -> impl Iterator<Item = &Attribute> {
    self.node.attrs.iter()
  }

  pub fn arg_types(&self) -> impl Iterator<Item = &Type> {
    self.node.types.iter().map(|n| &n.node.ty)
  }
}
