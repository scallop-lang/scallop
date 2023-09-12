use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
pub enum Item {
  ImportDecl(ImportDecl),
  TypeDecl(TypeDecl),
  ConstDecl(ConstDecl),
  RelationDecl(RelationDecl),
  QueryDecl(QueryDecl),
}

impl Item {
  pub fn attrs(&self) -> &Attributes {
    match self {
      Self::ImportDecl(i) => i.attrs(),
      Self::TypeDecl(t) => t.attrs(),
      Self::ConstDecl(c) => c.attrs(),
      Self::RelationDecl(r) => r.attrs(),
      Self::QueryDecl(q) => q.attrs(),
    }
  }

  pub fn attrs_mut(&mut self) -> &mut Attributes {
    match self {
      Self::ImportDecl(i) => i.attrs_mut(),
      Self::TypeDecl(t) => t.attrs_mut(),
      Self::ConstDecl(c) => c.attrs_mut(),
      Self::RelationDecl(r) => r.attrs_mut(),
      Self::QueryDecl(q) => q.attrs_mut(),
    }
  }

  pub fn is_query(&self) -> bool {
    match self {
      Self::QueryDecl(_) => true,
      _ => false,
    }
  }
}

pub type Items = Vec<Item>;
