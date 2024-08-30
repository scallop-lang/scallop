use crate::compiler::front::*;

#[derive(Debug)]
pub struct TransformTaggedRule<'a> {
  pub tagged_rule_analysis: &'a mut analyzers::TaggedRuleAnalysis,
}

impl<'a> TransformTaggedRule<'a> {
  pub fn new(tagged_rule_analysis: &'a mut analyzers::TaggedRuleAnalysis) -> Self {
    Self { tagged_rule_analysis }
  }

  pub fn has_prob_attr(rule_decl: &RuleDecl) -> bool {
    rule_decl.attrs().iter().find(|a| a.name().name() == "tagged").is_some()
  }
}

impl<'a> NodeVisitor<RuleDecl> for TransformTaggedRule<'a> {
  fn visit_mut(&mut self, rule_decl: &mut RuleDecl) {
    // If rule is directly declared with probability
    if let Some(tag) = rule_decl.tag().clone() {
      // Transform the rule
      let pred = rule_decl.rule_tag_predicate();

      // We create a new variable to hold the tag
      let tag_var_name = format!("{pred}#var");
      let tag_var = Variable::new(Identifier::new(tag_var_name.clone()));
      let tag_var_expr = Expr::variable(tag_var);

      // We generate a constraint encoding that `$variable == $tag`
      let eq_constraint = Formula::constraint(Constraint::new(Expr::binary(BinaryExpr::new(
        BinaryOp::new_eq(),
        tag_var_expr.clone(),
        tag.clone(),
      ))));

      // Generate the foreign predicate atom with that tag variable as the only argument
      let atom = Atom::new(Identifier::new(pred.clone()), vec![], vec![tag_var_expr]);
      let atom_formula = Formula::atom(atom);

      // Generate a formula that is the conjunction of constraint and atom
      let to_add_formula = Formula::conjunction(Conjunction::new(vec![eq_constraint, atom_formula]));

      // Update the original rule body
      let new_body = Formula::Conjunction(Conjunction::new(vec![to_add_formula, rule_decl.rule().body().clone()]));
      *rule_decl.rule_mut().body_mut() = new_body;

      // Remove the rule tag surface syntax
      *rule_decl.tag_mut() = None;

      // Tell the analyzer to store the information
      let rule_id = rule_decl.rule().location().clone();
      self
        .tagged_rule_analysis
        .add_tag_predicate(rule_id, pred, tag_var_name, tag.location().clone());
    } else if Self::has_prob_attr(rule_decl) {
      // Handle rules with external probabilities

      // If the rule is annotated with `@tagged`, we simply append a nullary atom at the end.
      // The fact will be populated by external sources.
      let pred = rule_decl.rule_tag_predicate();
      let new_atom = Formula::Atom(Atom::new(Identifier::new(pred.clone()), vec![], vec![]));
      let new_body = Formula::Conjunction(Conjunction::new(vec![new_atom, rule_decl.rule().body().clone()]));
      *rule_decl.rule_mut().body_mut() = new_body;
    }
  }
}
