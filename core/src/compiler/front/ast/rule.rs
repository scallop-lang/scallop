use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct RuleNode {
  pub head: Atom,
  pub body: Formula,
}

pub type Rule = AstNode<RuleNode>;

impl Rule {
  pub fn head(&self) -> &Atom {
    &self.node.head
  }

  pub fn body(&self) -> &Formula {
    &self.node.body
  }
}

impl Into<Vec<Item>> for Rule {
  fn into(self) -> Vec<Item> {
    vec![Item::RelationDecl(
      RelationDeclNode::Rule(
        RuleDeclNode {
          attrs: Attributes::new(),
          tag: Tag::default_none(),
          rule: self,
        }
        .into(),
      )
      .into(),
    )]
  }
}
