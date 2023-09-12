use super::*;

// / A single constant assignment, e.g. `X = 42`

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ConstAssignment {
  pub name: Identifier,
  pub ty: Option<Type>,
  pub value: Entity,
}

impl ConstAssignment {
  pub fn variable_name(&self) -> &String {
    self.name().name()
  }
}

// A (series of) constant declaration, e.g. `const X = 42, Y = true`
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ConstDecl {
  pub attrs: Attributes,
  pub assignments: Vec<ConstAssignment>,
}
