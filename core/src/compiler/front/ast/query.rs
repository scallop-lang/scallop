use super::*;

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub enum QueryNode {
  Predicate(Identifier),
  Atom(Atom),
}

pub type Query = AstNode<QueryNode>;

impl Query {
  pub fn predicate(&self) -> String {
    match &self.node {
      QueryNode::Predicate(p) => {
        let n = p.name();
        if let Some(id) = n.find("(") {
          n[..id].to_string()
        } else {
          n.to_string()
        }
      },
      QueryNode::Atom(a) => a.predicate().to_string(),
    }
  }

  pub fn create_relation_name(&self) -> String {
    match &self.node {
      QueryNode::Predicate(p) => p.name().to_string(),
      QueryNode::Atom(a) => format!("{}", a),
    }
  }
}

impl Into<Vec<Item>> for Query {
  fn into(self) -> Vec<Item> {
    vec![Item::QueryDecl(
      QueryDeclNode {
        attrs: Attributes::new(),
        query: self,
      }
      .into(),
    )]
  }
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct QueryDeclNode {
  pub attrs: Attributes,
  pub query: Query,
}

pub type QueryDecl = AstNode<QueryDeclNode>;

impl QueryDecl {
  pub fn attributes(&self) -> &Attributes {
    &self.node.attrs
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    &mut self.node.attrs
  }

  pub fn query(&self) -> &Query {
    &self.node.query
  }
}
