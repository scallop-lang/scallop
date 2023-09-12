use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub enum Query {
  Predicate(Identifier),
  Atom(Atom),
}

impl Query {
  pub fn formatted_predicate(&self) -> String {
    match self {
      Query::Predicate(p) => {
        let n = p.name();
        if let Some(id) = n.find("(") {
          n[..id].to_string()
        } else {
          n.to_string()
        }
      }
      Query::Atom(a) => a.predicate().to_string(),
    }
  }

  pub fn create_relation_name(&self) -> String {
    match self {
      Query::Predicate(p) => p.name().to_string(),
      Query::Atom(a) => format!("{}", a),
    }
  }
}

impl Into<Vec<Item>> for Query {
  fn into(self) -> Vec<Item> {
    vec![Item::QueryDecl(QueryDecl::new(Attributes::new(), self))]
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _QueryDecl {
  pub attrs: Attributes,
  pub query: Query,
}
