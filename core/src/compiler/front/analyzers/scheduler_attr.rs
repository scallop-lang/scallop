use std::collections::*;

use crate::compiler::front::*;
use crate::runtime::env::Scheduler;

#[derive(Clone, Debug)]
pub struct SchedulerAttributeAnalysis {
  pub scheduler_attrs: HashMap<String, Scheduler>,
  pub errors: Vec<FrontCompileErrorMessage>,
}

impl SchedulerAttributeAnalysis {
  pub fn new() -> Self {
    Self {
      scheduler_attrs: HashMap::new(),
      errors: Vec::new(),
    }
  }

  pub fn get_scheduler(&self, rel: &str) -> Option<&Scheduler> {
    self.scheduler_attrs.get(rel)
  }

  pub fn find_scheduler_attr<'a>(&mut self, attrs: &'a Vec<Attribute>) -> Option<Scheduler> {
    if let Some(attr) = attrs.iter().find(|attr| attr.attr_name() == "scheduler") {
      if let Some(scheduler_arg) = attr.pos_arg(0) {
        match scheduler_arg.as_string() {
          Some(scheduler_name) => {
            let scheduler = match scheduler_name.as_str() {
              "lfp" => Scheduler::LFP,
              "dfs" => Scheduler::DFS,
              "astar" => Scheduler::AStar,
              "beam" => {
                let maybe_beam_size = attr.kw_arg("beam_size").and_then(|av| av.as_integer()).map(|i| i as usize);
                let beam_size = maybe_beam_size.unwrap_or(3);
                Scheduler::Beam { beam_size }
              }
              n => {
                self.errors.push(FrontCompileErrorMessage::error()
                  .msg(&format!("Unknown scheduler `{}` for @scheduler attribute", n))
                  .src(scheduler_arg.location().clone()));
                return None;
              }
            };

            return Some(scheduler);
          }
          None => {
            self.errors.push(FrontCompileErrorMessage::error()
              .msg(&format!("Expected a string for the first positional argument for @scheduler"))
              .src(attr.location().clone()));
            return None;
          }
        }
      } else {
        self.errors.push(FrontCompileErrorMessage::error()
          .msg(&format!("Needs at least one positional argument for @scheduler"))
          .src(attr.location().clone()));
        return None;
      }
    } else {
      return None;
    }
  }
}

impl NodeVisitor<RuleDecl> for SchedulerAttributeAnalysis {
  fn visit(&mut self, node: &RuleDecl) {
    if let Some(scheduler) = self.find_scheduler_attr(node.attrs()) {
      for pred in node.rule().head().iter_predicates() {
        if !self.scheduler_attrs.contains_key(&pred) {
          self.scheduler_attrs.insert(pred.clone(), scheduler.clone());
        } else {
          self.errors.push(FrontCompileErrorMessage::error()
            .msg(&format!("@scheduler double specified on relation `{}`", pred))
            .src(node.location().clone()));
          return;
        }
      }
    }
  }
}

impl NodeVisitor<RelationTypeDecl> for SchedulerAttributeAnalysis {
  fn visit(&mut self, node: &RelationTypeDecl) {
    if let Some(scheduler) = self.find_scheduler_attr(node.attrs()) {
      for rel_type in node.rel_types() {
        let pred = rel_type.predicate_name();
        if !self.scheduler_attrs.contains_key(pred) {
          self.scheduler_attrs.insert(pred.clone(), scheduler.clone());
        } else {
          self.errors.push(FrontCompileErrorMessage::error()
            .msg(&format!("@scheduler double specified on relation `{}`", pred))
            .src(node.location().clone()));
          return;
        }
      }
    }
  }
}
