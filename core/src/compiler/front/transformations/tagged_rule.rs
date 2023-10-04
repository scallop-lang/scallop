use crate::common::input_tag::*;
use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub struct TransformTaggedRule {
  pub to_add_tags: Vec<(String, DynamicInputTag)>,
}

impl TransformTaggedRule {
  pub fn new() -> Self {
    Self { to_add_tags: vec![] }
  }

  pub fn has_prob_attr(rule_decl: &RuleDecl) -> bool {
    rule_decl
      .attrs()
      .iter()
      .find(|a| a.name().name() == "probabilistic")
      .is_some()
  }

  pub fn transform(rule_decl: &mut RuleDecl) -> String {
    // 1. Generate the predicate
    let pred = rule_decl.rule_tag_predicate();

    // 2. Append the atom to the end
    let new_atom = Formula::Atom(Atom::new(Identifier::new(pred.clone()), vec![], vec![]));
    let new_body = Formula::Conjunction(Conjunction::new(vec![new_atom, rule_decl.rule().body().clone()]));
    *rule_decl.rule_mut().body_mut() = new_body;

    // Return the predicate
    pred
  }

  pub fn drain_items(self) -> Vec<Item> {
    self
      .to_add_tags
      .into_iter()
      .map(|(pred, tag)| {
        let fact = Atom::new(Identifier::new(pred.clone()), vec![], vec![]);
        let fact_decl = FactDecl::new(vec![], Tag::new(tag), fact);
        let rel_decl = RelationDecl::Fact(fact_decl);
        let item = Item::RelationDecl(rel_decl);
        item
      })
      .collect()
  }
}

impl NodeVisitor<RuleDecl> for TransformTaggedRule {
  fn visit_mut(&mut self, rule_decl: &mut RuleDecl) {
    // If rule is directly declared with probability
    if rule_decl.tag().is_some() {
      // Transform the rule
      let pred = Self::transform(rule_decl);

      // Store this probability for later
      self.to_add_tags.push((pred.clone(), rule_decl.tag().tag().clone()));
    } else if Self::has_prob_attr(rule_decl) {
      // If the rule is annotated with `@probabilistic`
      Self::transform(rule_decl);
    }
  }
}
