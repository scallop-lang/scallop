use super::*;

#[derive(Clone, Debug, PartialEq)]
pub enum Item {
  ImportDecl(ImportDecl),
  InputDecl(InputDecl),
  TypeDecl(TypeDecl),
  RelationDecl(RelationDecl),
  QueryDecl(QueryDecl),
}

impl Item {
  pub fn attributes(&self) -> &Attributes {
    match self {
      Self::ImportDecl(i) => i.attributes(),
      Self::InputDecl(i) => i.attributes(),
      Self::TypeDecl(t) => t.attributes(),
      Self::RelationDecl(r) => r.attributes(),
      Self::QueryDecl(q) => q.attributes(),
    }
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    match self {
      Self::ImportDecl(i) => i.attributes_mut(),
      Self::InputDecl(i) => i.attributes_mut(),
      Self::TypeDecl(t) => t.attributes_mut(),
      Self::RelationDecl(r) => r.attributes_mut(),
      Self::QueryDecl(q) => q.attributes_mut(),
    }
  }
}

impl WithLocation for Item {
  fn location(&self) -> &AstNodeLocation {
    match self {
      Self::ImportDecl(i) => i.location(),
      Self::InputDecl(i) => i.location(),
      Self::TypeDecl(t) => t.location(),
      Self::RelationDecl(r) => r.location(),
      Self::QueryDecl(q) => q.location(),
    }
  }
}

pub type Items = Vec<Item>;
