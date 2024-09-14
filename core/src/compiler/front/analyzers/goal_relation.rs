use std::collections::*;

use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub struct GoalRelationAnalysis {
  pub goal_relations: HashMap<String, NodeLocation>,
  pub errors: Vec<FrontCompileErrorMessage>,
}

impl GoalRelationAnalysis {
  pub fn new() -> Self {
    Self {
      goal_relations: HashMap::new(),
      errors: vec![],
    }
  }

  pub fn is_goal(&self, rel: &str) -> bool {
    self.goal_relations.contains_key(rel)
  }

  pub fn get_goal_attr<'a>(&self, attrs: &'a Vec<Attribute>) -> Option<&'a Attribute> {
    attrs.iter().find(|attr| attr.attr_name() == "goal")
  }

  pub fn compute_errors(&mut self) {
    // Check if there is more than one relation that is annotated with `@goal`
    if self.goal_relations.len() > 1 {
      let mut iterator = self.goal_relations.iter();
      let (first_pred, first_loc) = iterator.next().unwrap();
      let (second_pred, second_loc) = iterator.next().unwrap(); // both unwrap will work because the size >= 2
      self.errors.push(
        FrontCompileErrorMessage::error()
          .msg(format!(
            "There are more than one relations that are annotated with @goal, where there could be at most 1. For instance, one of them is `{}`:",
            first_pred,
          ))
          .src(first_loc.clone())
          .msg(format!(
            "and the second is `{}`:",
            second_pred,
          ))
          .src(second_loc.clone())
      )
    }
  }

  pub fn add_incorrect_relation_arity_error(&mut self, pred: &String, num_args: usize, loc: &NodeLocation) {
    self.errors.push(
      FrontCompileErrorMessage::error()
        .msg(format!(
          "@goal annotated relation must be of arity-0; however we find the relation `{}` to have arity {}",
          pred, num_args,
        ))
        .src(loc.clone()),
    )
  }

  pub fn add_goal_relation(&mut self, pred: &String, loc: &NodeLocation) {
    self.goal_relations.insert(pred.clone(), loc.clone());
  }
}

impl NodeVisitor<RuleDecl> for GoalRelationAnalysis {
  fn visit(&mut self, node: &RuleDecl) {
    if let Some(_) = self.get_goal_attr(node.attrs()) {
      match node.rule().head() {
        RuleHead::Atom(atom) => {
          if atom.num_args() != 0 {
            self.add_incorrect_relation_arity_error(atom.predicate().name(), atom.num_args(), atom.location())
          } else {
            self.add_goal_relation(atom.predicate().name(), atom.location())
          }
        }
        _ => self.errors.push(
          FrontCompileErrorMessage::error()
            .msg(format!(
              "@goal annotated rule cannot have a head other than simple atom"
            ))
            .src(node.rule().head().location().clone()),
        ),
      }
    }
  }
}

impl NodeVisitor<RelationTypeDecl> for GoalRelationAnalysis {
  fn visit(&mut self, node: &RelationTypeDecl) {
    if let Some(_) = self.get_goal_attr(node.attrs()) {
      for relation_type in node.rel_types() {
        if relation_type.num_arg_bindings() != 0 {
          self.add_incorrect_relation_arity_error(
            relation_type.predicate_name(),
            relation_type.num_arg_bindings(),
            relation_type.location(),
          )
        } else {
          self.add_goal_relation(relation_type.predicate_name(), relation_type.location())
        }
      }
    }
  }
}
