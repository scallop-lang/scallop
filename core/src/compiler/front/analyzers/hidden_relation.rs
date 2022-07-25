use std::collections::*;

use super::super::*;

#[derive(Clone, Debug)]
pub struct HiddenRelationAnalysis {
  pub hidden_relations: HashSet<String>,
}

impl HiddenRelationAnalysis {
  pub fn new() -> Self {
    Self {
      hidden_relations: HashSet::new(),
    }
  }

  pub fn contains(&self, rela: &String) -> bool {
    self.hidden_relations.contains(rela)
  }

  pub fn process_attributes(&mut self, pred: &str, attrs: &Attributes) {
    if attrs.iter().find(|a| a.name() == "hidden").is_some() {
      self.hidden_relations.insert(pred.to_string());
    }
  }
}

impl NodeVisitor for HiddenRelationAnalysis {
  fn visit_input_decl(&mut self, input_decl: &InputDecl) {
    self.process_attributes(input_decl.predicate(), input_decl.attributes())
  }

  fn visit_relation_type_decl(&mut self, rela_type_decl: &ast::RelationTypeDecl) {
    self.process_attributes(rela_type_decl.predicate(), rela_type_decl.attributes())
  }

  fn visit_constant_set_decl(&mut self, decl: &ast::ConstantSetDecl) {
    self.process_attributes(decl.predicate(), decl.attributes())
  }

  fn visit_fact_decl(&mut self, decl: &ast::FactDecl) {
    self.process_attributes(decl.predicate(), decl.attributes())
  }

  fn visit_rule_decl(&mut self, rule_decl: &RuleDecl) {
    self.process_attributes(rule_decl.rule().head().predicate(), rule_decl.attributes())
  }
}
