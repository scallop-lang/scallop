use super::*;

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub enum TypeDeclNode {
  Subtype(SubtypeDecl),
  Alias(AliasTypeDecl),
  Relation(RelationTypeDecl),
}

pub type TypeDecl = AstNode<TypeDeclNode>;

impl TypeDecl {
  pub fn attributes(&self) -> &Attributes {
    match &self.node {
      TypeDeclNode::Subtype(s) => s.attributes(),
      TypeDeclNode::Alias(a) => a.attributes(),
      TypeDeclNode::Relation(r) => r.attributes(),
    }
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    match &mut self.node {
      TypeDeclNode::Subtype(s) => s.attributes_mut(),
      TypeDeclNode::Alias(a) => a.attributes_mut(),
      TypeDeclNode::Relation(r) => r.attributes_mut(),
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct SubtypeDeclNode {
  pub attrs: Attributes,
  pub name: Identifier,
  pub subtype_of: Type,
}

pub type SubtypeDecl = AstNode<SubtypeDeclNode>;

impl SubtypeDecl {
  pub fn name(&self) -> &str {
    self.node.name.name()
  }

  pub fn subtype_of(&self) -> &Type {
    &self.node.subtype_of
  }

  pub fn attributes(&self) -> &Attributes {
    &self.node.attrs
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    &mut self.node.attrs
  }
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct AliasTypeDeclNode {
  pub attrs: Attributes,
  pub name: Identifier,
  pub alias_of: Type,
}

pub type AliasTypeDecl = AstNode<AliasTypeDeclNode>;

impl AliasTypeDecl {
  pub fn name(&self) -> &str {
    self.node.name.name()
  }

  pub fn alias_of(&self) -> &Type {
    &self.node.alias_of
  }

  pub fn attributes(&self) -> &Attributes {
    &self.node.attrs
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    &mut self.node.attrs
  }
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct ArgTypeBindingNode {
  pub name: Option<Identifier>,
  pub ty: Type,
}

pub type ArgTypeBinding = AstNode<ArgTypeBindingNode>;

impl ArgTypeBinding {
  pub fn name(&self) -> Option<&str> {
    self.node.name.as_ref().map(Identifier::name)
  }

  pub fn ty(&self) -> &Type {
    &self.node.ty
  }
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct RelationTypeNode {
  pub name: Identifier,
  pub arg_types: Vec<ArgTypeBinding>,
}

pub type RelationType = AstNode<RelationTypeNode>;

impl Into<Vec<Item>> for RelationType {
  fn into(self) -> Vec<Item> {
    vec![Item::TypeDecl(
      TypeDeclNode::Relation(
        RelationTypeDeclNode {
          attrs: Attributes::new(),
          rel_types: vec![self],
        }
        .into(),
      )
      .into(),
    )]
  }
}

impl RelationType {
  pub fn predicate(&self) -> &str {
    self.node.name.name()
  }

  pub fn arg_types(&self) -> impl Iterator<Item = &Type> {
    self.node.arg_types.iter().map(|arg| arg.ty())
  }
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct RelationTypeDeclNode {
  pub attrs: Attributes,
  pub rel_types: Vec<RelationType>,
}

pub type RelationTypeDecl = AstNode<RelationTypeDeclNode>;

impl RelationTypeDecl {
  pub fn relation_types(&self) -> impl Iterator<Item = &RelationType> {
    self.node.rel_types.iter()
  }

  pub fn relation_types_mut(&mut self) -> impl Iterator<Item = &mut RelationType> {
    self.node.rel_types.iter_mut()
  }

  pub fn predicates(&self) -> impl Iterator<Item = &str> {
    self.relation_types().map(|rel_type| rel_type.predicate())
  }

  pub fn attributes(&self) -> &Vec<Attribute> {
    &self.node.attrs
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    &mut self.node.attrs
  }
}
