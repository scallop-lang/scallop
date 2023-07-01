use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub struct TransformConjunctiveHead {
  to_add_items: Vec<Item>,
}

impl TransformConjunctiveHead {
  pub fn new() -> Self {
    Self { to_add_items: vec![] }
  }

  pub fn retain(&self, item: &Item) -> bool {
    match item {
      Item::RelationDecl(r) => {
        if let Some(rule) = r.rule() {
          !rule.head().is_conjunction()
        } else {
          true
        }
      }
      _ => true,
    }
  }

  pub fn generate_items(self) -> Vec<Item> {
    self.to_add_items
  }
}

impl NodeVisitorMut for TransformConjunctiveHead {
  fn visit_rule(&mut self, rule: &mut Rule) {
    match &rule.head().node {
      RuleHeadNode::Conjunction(c) => {
        for atom in c {
          self.to_add_items.push(Item::RelationDecl(
            RelationDeclNode::Rule(
              RuleDeclNode {
                attrs: Attributes::new(),
                tag: Tag::default_none(),
                rule: Rule::new(
                  rule.location().clone_without_id(),
                  RuleNode {
                    head: RuleHead::new(rule.location().clone(), RuleHeadNode::Atom(atom.clone())),
                    body: rule.body().clone(),
                  },
                ),
              }
              .into(),
            )
            .into(),
          ));
        }
      }
      _ => {}
    }
  }
}
