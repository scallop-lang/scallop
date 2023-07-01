use serde::*;

use super::*;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[doc(hidden)]
pub enum TypeDeclNode {
  Subtype(SubtypeDecl),
  Alias(AliasTypeDecl),
  Relation(RelationTypeDecl),
  Enum(EnumTypeDecl),
  Algebraic(AlgebraicDataTypeDecl),
}

pub type TypeDecl = AstNode<TypeDeclNode>;

impl TypeDecl {
  pub fn alias(name: Identifier, alias_of: Type) -> Self {
    TypeDeclNode::Alias(
      AliasTypeDeclNode {
        attrs: Attributes::default(),
        name,
        alias_of,
      }
      .into(),
    )
    .into()
  }

  pub fn relation(name: Identifier, args: Vec<Type>) -> Self {
    TypeDeclNode::Relation(
      RelationTypeDeclNode {
        attrs: Attributes::default(),
        rel_types: vec![RelationTypeNode {
          name,
          arg_types: args
            .into_iter()
            .map(|a| ArgTypeBindingNode { name: None, ty: a }.into())
            .collect(),
        }
        .into()],
      }
      .into(),
    )
    .into()
  }

  pub fn attributes(&self) -> &Attributes {
    match &self.node {
      TypeDeclNode::Subtype(s) => s.attributes(),
      TypeDeclNode::Alias(a) => a.attributes(),
      TypeDeclNode::Relation(r) => r.attributes(),
      TypeDeclNode::Enum(e) => e.attributes(),
      TypeDeclNode::Algebraic(a) => a.attributes(),
    }
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    match &mut self.node {
      TypeDeclNode::Subtype(s) => s.attributes_mut(),
      TypeDeclNode::Alias(a) => a.attributes_mut(),
      TypeDeclNode::Relation(r) => r.attributes_mut(),
      TypeDeclNode::Enum(e) => e.attributes_mut(),
      TypeDeclNode::Algebraic(a) => a.attributes_mut(),
    }
  }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
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

#[derive(Clone, Debug, PartialEq, Serialize)]
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

#[derive(Clone, Debug, PartialEq, Serialize)]
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

#[derive(Clone, Debug, PartialEq, Serialize)]
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

#[derive(Clone, Debug, PartialEq, Serialize)]
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

#[derive(Clone, Debug, PartialEq, Serialize)]
#[doc(hidden)]
pub struct EnumTypeDeclNode {
  pub attrs: Attributes,
  pub name: Identifier,
  pub members: Vec<EnumTypeMember>,
}

pub type EnumTypeDecl = AstNode<EnumTypeDeclNode>;

impl EnumTypeDecl {
  pub fn attributes(&self) -> &Vec<Attribute> {
    &self.node.attrs
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    &mut self.node.attrs
  }

  pub fn name(&self) -> &str {
    self.node.name.name()
  }

  pub fn members(&self) -> &[EnumTypeMember] {
    &self.node.members
  }

  pub fn iter_members(&self) -> impl Iterator<Item = &EnumTypeMember> {
    self.node.members.iter()
  }

  pub fn iter_members_mut(&mut self) -> impl Iterator<Item = &mut EnumTypeMember> {
    self.node.members.iter_mut()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EnumTypeMemberNode {
  pub name: Identifier,
  pub assigned_num: Option<Constant>,
}

pub type EnumTypeMember = AstNode<EnumTypeMemberNode>;

impl EnumTypeMember {
  pub fn name(&self) -> &str {
    self.node.name.name()
  }

  pub fn assigned_number(&self) -> Option<&Constant> {
    self.node.assigned_num.as_ref()
  }

  pub fn assigned_number_mut(&mut self) -> Option<&mut Constant> {
    self.node.assigned_num.as_mut()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AlgebraicDataTypeDeclNode {
  pub attrs: Attributes,
  pub name: Identifier,
  pub variants: Vec<AlgebraicDataTypeVariant>,
}

pub type AlgebraicDataTypeDecl = AstNode<AlgebraicDataTypeDeclNode>;

impl AlgebraicDataTypeDecl {
  pub fn attributes(&self) -> &Vec<Attribute> {
    &self.node.attrs
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    &mut self.node.attrs
  }

  pub fn name(&self) -> &str {
    self.node.name.name()
  }

  pub fn name_identifier(&self) -> &Identifier {
    &self.node.name
  }

  pub fn iter_variants(&self) -> impl Iterator<Item = &AlgebraicDataTypeVariant> {
    self.node.variants.iter()
  }

  pub fn iter_variants_mut(&mut self) -> impl Iterator<Item = &mut AlgebraicDataTypeVariant> {
    self.node.variants.iter_mut()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AlgebraicDataTypeVariantNode {
  pub constructor: Identifier,
  pub args: Vec<Type>,
}

pub type AlgebraicDataTypeVariant = AstNode<AlgebraicDataTypeVariantNode>;

impl AlgebraicDataTypeVariant {
  pub fn name(&self) -> &str {
    self.node.constructor.name()
  }

  pub fn name_identifier(&self) -> &Identifier {
    &self.node.constructor
  }

  pub fn iter_arg_types(&self) -> impl Iterator<Item = &Type> {
    self.node.args.iter()
  }

  pub fn iter_arg_types_mut(&mut self) -> impl Iterator<Item = &mut Type> {
    self.node.args.iter_mut()
  }

  pub fn args(&self) -> &Vec<Type> {
    &self.node.args
  }
}
