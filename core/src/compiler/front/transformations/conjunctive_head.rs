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
        if let Some(rule) = r.as_rule() {
          !rule.rule().head().is_conjunction()
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

impl NodeVisitor<Rule> for TransformConjunctiveHead {
  fn visit_mut(&mut self, rule: &mut Rule) {
    match rule.head() {
      RuleHead::Conjunction(c) => {
        for atom in c.iter_atoms() {
          self.to_add_items.push(
            Item::RelationDecl(
              RelationDecl::Rule(
                RuleDecl::new(
                  Attributes::new(),
                  Tag::none(),
                  Rule::new_with_loc(
                    RuleHead::atom(atom.clone()),
                    rule.body().clone(),
                    rule.location().clone_without_id(),
                  ),
                )
              ),
            ),
          );
        }
      }
      _ => {}
    }
  }
}
