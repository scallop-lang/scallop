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

  pub fn contains(&self, rela: &str) -> bool {
    self.hidden_relations.contains(rela)
  }

  pub fn process_attributes(&mut self, pred: &str, attrs: &Attributes) {
    if attrs.find("hidden").is_some() {
      self.hidden_relations.insert(pred.to_string());
    }
  }
}

impl NodeVisitor<RelationTypeDecl> for HiddenRelationAnalysis {
  fn visit(&mut self, rela_type_decl: &RelationTypeDecl) {
    for rela_type in rela_type_decl.rel_types() {
      self.process_attributes(rela_type.predicate_name(), rela_type_decl.attrs());
    }
  }
}

impl NodeVisitor<ConstantSetDecl> for HiddenRelationAnalysis {
  fn visit(&mut self, decl: &ConstantSetDecl) {
    self.process_attributes(decl.predicate_name(), decl.attrs())
  }
}

impl NodeVisitor<FactDecl> for HiddenRelationAnalysis {
  fn visit(&mut self, decl: &FactDecl) {
    self.process_attributes(decl.predicate_name(), decl.attrs())
  }
}

impl NodeVisitor<RuleDecl> for HiddenRelationAnalysis {
  fn visit(&mut self, rule_decl: &RuleDecl) {
    for predicate in rule_decl.rule().head().iter_predicates() {
      self.process_attributes(&predicate, rule_decl.attrs())
    }
  }
}
