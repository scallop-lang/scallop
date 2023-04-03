use std::collections::*;

use crate::common::foreign_predicate::*;

use super::super::utils::*;
use super::super::*;

#[derive(Clone, Debug)]
pub struct HeadRelationAnalysis {
  pub errors: Vec<HeadRelationError>,
  pub used_relations: HashMap<String, Loc>,
  pub declared_relations: HashSet<String>,
}

impl HeadRelationAnalysis {
  pub fn new(foreign_predicate_registry: &ForeignPredicateRegistry) -> Self {
    let declared_relations = foreign_predicate_registry.iter().map(|(_, p)| p.name().to_string()).collect();
    Self {
      errors: vec![],
      used_relations: HashMap::new(),
      declared_relations,
    }
  }

  pub fn add_foreign_predicate<F: ForeignPredicate>(&mut self, fp: &F) {
    self.declared_relations.insert(fp.name().to_string());
  }

  pub fn compute_errors(&mut self) {
    let used_relations_set = self.used_relations.keys().cloned().collect::<HashSet<String>>();
    for r in used_relations_set.difference(&self.declared_relations) {
      if !r.contains("#") {
        self.errors.push(HeadRelationError::RelationNotInHeadWarning {
          relation: r.clone(),
          occurred: self.used_relations[r].clone(),
        });
      }
    }
  }
}

impl NodeVisitor for HeadRelationAnalysis {
  fn visit_relation_type(&mut self, rd: &ast::RelationType) {
    self.declared_relations.insert(rd.predicate().to_string());
  }

  fn visit_fact_decl(&mut self, fd: &ast::FactDecl) {
    self.declared_relations.insert(fd.predicate().to_string());
  }

  fn visit_constant_set_decl(&mut self, csd: &ast::ConstantSetDecl) {
    self.declared_relations.insert(csd.predicate().to_string());
  }

  fn visit_rule(&mut self, rd: &ast::Rule) {
    self.declared_relations.insert(rd.head().predicate().to_string());
  }

  fn visit_query(&mut self, qd: &ast::Query) {
    self
      .used_relations
      .insert(qd.create_relation_name().to_string(), qd.location().clone());
  }

  fn visit_atom(&mut self, a: &ast::Atom) {
    self
      .used_relations
      .insert(a.predicate().to_string(), a.location().clone());
  }
}

#[derive(Debug, Clone)]
pub enum HeadRelationError {
  RelationNotInHeadWarning { relation: String, occurred: Loc },
}

impl FrontCompileErrorTrait for HeadRelationError {
  fn error_type(&self) -> FrontCompileErrorType {
    match self {
      Self::RelationNotInHeadWarning { .. } => FrontCompileErrorType::Warning,
    }
  }

  fn report(&self, src: &Sources) -> String {
    match self {
      Self::RelationNotInHeadWarning { relation, occurred } => {
        format!(
          "relation `{}` is not computed but directly used; consider adding a type declaration: \n{}",
          relation,
          occurred.report_warning(src)
        )
      }
    }
  }
}
