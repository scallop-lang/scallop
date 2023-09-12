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
    let declared_relations = foreign_predicate_registry
      .iter()
      .map(|(_, p)| p.name().to_string())
      .collect();
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

impl NodeVisitor<RelationType> for HeadRelationAnalysis {
  fn visit(&mut self, rd: &RelationType) {
    self.declared_relations.insert(rd.predicate_name().to_string());
  }
}

impl NodeVisitor<FactDecl> for HeadRelationAnalysis {
  fn visit(&mut self, fd: &FactDecl) {
    self.declared_relations.insert(fd.predicate_name().to_string());
  }
}

impl NodeVisitor<ConstantSetDecl> for HeadRelationAnalysis {
  fn visit(&mut self, csd: &ConstantSetDecl) {
    self.declared_relations.insert(csd.name().to_string());
  }
}

impl NodeVisitor<Rule> for HeadRelationAnalysis {
  fn visit(&mut self, rd: &Rule) {
    for predicate in rd.head().iter_predicates() {
      self.declared_relations.insert(predicate.to_string());
    }
  }
}

impl NodeVisitor<Query> for HeadRelationAnalysis {
  fn visit(&mut self, qd: &Query) {
    self
      .used_relations
      .insert(qd.create_relation_name().to_string(), qd.location().clone());
  }
}

impl NodeVisitor<Atom> for HeadRelationAnalysis {
  fn visit(&mut self, a: &Atom) {
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
