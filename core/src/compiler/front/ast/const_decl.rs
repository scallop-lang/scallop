use serde::*;

use super::*;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[doc(hidden)]
pub struct ConstAssignmentNode {
  pub name: Identifier,
  pub ty: Option<Type>,
  pub value: Entity,
}

/// A single constant assignment, e.g. `X = 42`
pub type ConstAssignment = AstNode<ConstAssignmentNode>;

impl ConstAssignment {
  pub fn identifier(&self) -> &Identifier {
    &self.node.name
  }

  pub fn identifier_mut(&mut self) -> &mut Identifier {
    &mut self.node.name
  }

  pub fn name(&self) -> &str {
    self.node.name.name()
  }

  pub fn ty(&self) -> Option<&Type> {
    self.node.ty.as_ref()
  }

  pub fn ty_mut(&mut self) -> Option<&mut Type> {
    self.node.ty.as_mut()
  }

  pub fn value(&self) -> &Entity {
    &self.node.value
  }

  pub fn value_mut(&mut self) -> &mut Entity {
    &mut self.node.value
  }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[doc(hidden)]
pub struct ConstDeclNode {
  pub attrs: Attributes,
  pub assignments: Vec<ConstAssignment>,
}

/// A (series of) constant declaration, e.g. `const X = 42`
pub type ConstDecl = AstNode<ConstDeclNode>;

impl ConstDecl {
  pub fn attributes(&self) -> &Attributes {
    &self.node.attrs
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    &mut self.node.attrs
  }

  pub fn iter_assignments(&self) -> impl Iterator<Item = &ConstAssignment> {
    self.node.assignments.iter()
  }

  pub fn iter_assignments_mut(&mut self) -> impl Iterator<Item = &mut ConstAssignment> {
    self.node.assignments.iter_mut()
  }
}
