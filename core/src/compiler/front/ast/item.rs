use super::*;

#[derive(Clone, Debug, PartialEq)]
pub enum Item {
  ImportDecl(ImportDecl),
  TypeDecl(TypeDecl),
  ConstDecl(ConstDecl),
  RelationDecl(RelationDecl),
  QueryDecl(QueryDecl),
}

impl Item {
  pub fn attributes(&self) -> &Attributes {
    match self {
      Self::ImportDecl(i) => i.attributes(),
      Self::TypeDecl(t) => t.attributes(),
      Self::ConstDecl(c) => c.attributes(),
      Self::RelationDecl(r) => r.attributes(),
      Self::QueryDecl(q) => q.attributes(),
    }
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    match self {
      Self::ImportDecl(i) => i.attributes_mut(),
      Self::TypeDecl(t) => t.attributes_mut(),
      Self::ConstDecl(c) => c.attributes_mut(),
      Self::RelationDecl(r) => r.attributes_mut(),
      Self::QueryDecl(q) => q.attributes_mut(),
    }
  }
}

impl WithLocation for Item {
  fn location(&self) -> &AstNodeLocation {
    match self {
      Self::ImportDecl(i) => i.location(),
      Self::TypeDecl(t) => t.location(),
      Self::ConstDecl(c) => c.location(),
      Self::RelationDecl(r) => r.location(),
      Self::QueryDecl(q) => q.location(),
    }
  }

  fn location_mut(&mut self) -> &mut AstNodeLocation {
    match self {
      Self::ImportDecl(i) => i.location_mut(),
      Self::TypeDecl(t) => t.location_mut(),
      Self::ConstDecl(c) => c.location_mut(),
      Self::RelationDecl(r) => r.location_mut(),
      Self::QueryDecl(q) => q.location_mut(),
    }
  }
}

pub type Items = Vec<Item>;
