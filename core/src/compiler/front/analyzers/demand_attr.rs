use std::collections::*;

use super::{super::*, type_inference};

#[derive(Clone, Debug)]
pub struct DemandAttributeAnalysis {
  pub demand_attrs: HashMap<String, (String, AstNodeLocation)>,
  pub errors: Vec<DemandAttributeError>,
}

impl DemandAttributeAnalysis {
  pub fn new() -> Self {
    Self {
      demand_attrs: HashMap::new(),
      errors: Vec::new(),
    }
  }

  pub fn demand_pattern(&self, pred: &String) -> Option<&String> {
    self.demand_attrs.get(pred).map(|(p, _)| p)
  }

  pub fn check_arity(&mut self, type_inference: &type_inference::TypeInference) {
    for (pred, (pattern, loc)) in &self.demand_attrs {
      let (tys, _) = type_inference.inferred_relation_types.get(pred).unwrap();
      if pattern.len() != tys.len() {
        self.errors.push(DemandAttributeError::ArityMismatch {
          pattern: pattern.clone(),
          expected: tys.len(),
          actual: pattern.len(),
          loc: loc.clone(),
        });
      }
    }
  }

  pub fn process_attribute(&mut self, pred: &str, attr: &Attribute) {
    if attr.name() == "demand" {
      if attr.num_pos_args() == 1 {
        let value = attr.pos_arg(0).unwrap();
        match &value.node {
          ConstantNode::String(s) => {
            if is_valid_demand_pattern(s) {
              if let Some((p, l)) = self.demand_attrs.get(pred) {
                if p != s {
                  self.errors.push(DemandAttributeError::ConflictingPattern {
                    first_loc: l.clone(),
                    second_loc: value.location().clone(),
                  });
                }
              } else {
                let attr = (s.clone(), value.location().clone());
                self.demand_attrs.insert(pred.to_string(), attr);
              }
            } else {
              self.errors.push(DemandAttributeError::InvalidPattern {
                loc: value.location().clone(),
              });
            }
          }
          _ => self.errors.push(DemandAttributeError::InvalidArgumentType {
            found: value.kind().to_string(),
            loc: value.location().clone(),
          }),
        }
      } else {
        self.errors.push(DemandAttributeError::InvalidNumArgs {
          pred: pred.to_string(),
          actual_num_args: attr.num_pos_args(),
          loc: attr.location().clone(),
        });
      }
    }
  }

  pub fn process_attributes(&mut self, pred: &str, attributes: &Attributes) {
    attributes.iter().for_each(|attr| {
      self.process_attribute(pred, attr);
    });
  }
}

impl NodeVisitor for DemandAttributeAnalysis {
  fn visit_relation_type_decl(&mut self, rel_type_decl: &ast::RelationTypeDecl) {
    self.process_attributes(rel_type_decl.predicate(), rel_type_decl.attributes());
  }

  fn visit_rule_decl(&mut self, rule_decl: &ast::RuleDecl) {
    self.process_attributes(rule_decl.rule().head().predicate(), rule_decl.attributes());
  }
}

fn is_valid_demand_pattern(pattern: &String) -> bool {
  pattern.chars().all(|c| c == 'b' || c == 'f')
}

#[derive(Clone, Debug)]
pub enum DemandAttributeError {
  InvalidNumArgs {
    pred: String,
    actual_num_args: usize,
    loc: AstNodeLocation,
  },
  InvalidArgumentType {
    found: String,
    loc: AstNodeLocation,
  },
  ConflictingPattern {
    first_loc: AstNodeLocation,
    second_loc: AstNodeLocation,
  },
  ArityMismatch {
    pattern: String,
    expected: usize,
    actual: usize,
    loc: AstNodeLocation,
  },
  InvalidPattern {
    loc: AstNodeLocation,
  },
}

impl From<DemandAttributeError> for FrontCompileError {
  fn from(e: DemandAttributeError) -> Self {
    Self::DemandAttributeError(e)
  }
}

impl DemandAttributeError {
  pub fn report(&self, src: &Sources) {
    match self {
      Self::InvalidNumArgs {
        pred,
        actual_num_args,
        loc,
      } => {
        println!(
          "Invalid number of arguments of @demand attribute for `{}`. Expected 1, Found {}",
          pred, actual_num_args
        );
        loc.report(src);
      }
      Self::InvalidArgumentType { found, loc } => {
        println!("Invalid argument type. Expected `string`, found `{}`", found);
        loc.report(src);
      }
      Self::ConflictingPattern { first_loc, second_loc } => {
        println!("Conflicting demand pattern. First defined here:");
        first_loc.report(src);
        println!("re-defined here:");
        second_loc.report(src);
      }
      Self::ArityMismatch {
        pattern,
        expected,
        actual,
        loc,
      } => {
        println!(
          "Arity mismatch for demand pattern `{}`. Expected {}, found {}",
          pattern, expected, actual
        );
        loc.report(src);
      }
      Self::InvalidPattern { loc } => {
        println!("Invalid demand pattern");
        loc.report(src);
      }
    }
  }
}
