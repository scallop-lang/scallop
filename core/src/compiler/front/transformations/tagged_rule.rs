use crate::{common::input_tag::InputTag, compiler::front::*};

#[derive(Clone, Debug)]
pub struct TransformTaggedRule {
  pub to_add_tags: Vec<(String, InputTag)>,
}

impl TransformTaggedRule {
  pub fn new() -> Self {
    Self {
      to_add_tags: vec![],
    }
  }

  pub fn has_prob_attr(rule_decl: &RuleDecl) -> bool {
    rule_decl
      .attributes()
      .iter()
      .find(|a| a.name() == "probabilistic")
      .is_some()
  }

  pub fn transform(rule_decl: &mut RuleDecl) -> String {
    // 1. Generate the predicate
    let pred = rule_decl.rule_tag_predicate();

    // 2. Append the atom to the end
    let new_atom = AtomNode {
      predicate: IdentifierNode { name: pred.clone() }.into(),
      args: vec![],
    };
    let new_atom_form = Formula::Atom(new_atom.into());
    let new_conj = ConjunctionNode {
      args: vec![new_atom_form, rule_decl.node.rule.node.body.clone()],
    };
    let new_body = Formula::Conjunction(new_conj.into());
    rule_decl.node.rule.node.body = new_body;

    // Return the predicate
    pred
  }
}

impl NodeVisitorMut for TransformTaggedRule {
  fn visit_rule_decl(&mut self, rule_decl: &mut RuleDecl) {
    // If rule is directly declared with probability
    if rule_decl.tag().is_some() {
      // Transform the rule
      let pred = Self::transform(rule_decl);

      // Store this probability for later
      self
        .to_add_tags
        .push((pred.clone(), rule_decl.tag().input_tag().clone()));
    } else if Self::has_prob_attr(rule_decl) {
      // If the rule is annotated with `@probabilistic`
      Self::transform(rule_decl);
    }
  }
}

impl Transformation for TransformTaggedRule {
  fn generate_items(self) -> Vec<Item> {
    self
      .to_add_tags
      .into_iter()
      .map(|(pred, tag)| {
        let fact = AtomNode {
          predicate: IdentifierNode { name: pred.clone() }.into(),
          args: vec![],
        };
        let fact_decl = FactDeclNode {
          attrs: vec![],
          tag: TagNode(tag).into(),
          atom: fact.into(),
        };
        let rel_decl = RelationDeclNode::Fact(fact_decl.into());
        let item = Item::RelationDecl(rel_decl.into());
        item
      })
      .collect()
  }
}
