use std::collections::*;

use super::super::*;
use super::type_inference;

#[derive(Clone, Debug)]
pub struct DemandAttributeAnalysis {
  pub demand_attrs: HashMap<String, (String, NodeLocation)>,
  pub disjunctive_predicates: HashSet<String>,
  pub errors: Vec<DemandAttributeError>,
}

impl DemandAttributeAnalysis {
  pub fn new() -> Self {
    Self {
      demand_attrs: HashMap::new(),
      disjunctive_predicates: HashSet::new(),
      errors: Vec::new(),
    }
  }

  pub fn demand_pattern(&self, pred: &String) -> Option<&String> {
    self.demand_attrs.get(pred).map(|(p, _)| p)
  }

  pub fn check_arity(&mut self, type_inference: &type_inference::TypeInference) {
    for (pred, (pattern, loc)) in &self.demand_attrs {
      if let Some((tys, _)) = type_inference.inferred_relation_types.get(pred) {
        if pattern.len() != tys.len() {
          self.errors.push(DemandAttributeError::ArityMismatch {
            pattern: pattern.clone(),
            expected: tys.len(),
            actual: pattern.len(),
            loc: loc.clone(),
          });
        }
      } else {
        // This means that there is an error from type inference; we do not handle it now
      }
    }
  }

  pub fn set_disjunctive(&mut self, pred: &String, loc: &NodeLocation) {
    if self.demand_attrs.contains_key(pred) {
      self
        .errors
        .push(DemandAttributeError::DisjunctivePredicateWithDemandAttribute {
          pred: pred.clone(),
          loc: loc.clone(),
        });
    } else {
      self.disjunctive_predicates.insert(pred.clone());
    }
  }

  pub fn process_attribute(&mut self, pred: &str, attr: &Attribute) {
    // Check if the predicate occurs in a disjunctive head
    if self.disjunctive_predicates.contains(pred) {
      self
        .errors
        .push(DemandAttributeError::DisjunctivePredicateWithDemandAttribute {
          pred: pred.to_string(),
          loc: attr.location().clone(),
        });
    }

    // Check the pattern
    if attr.name().name().as_str() == "demand" {
      if attr.num_pos_args() == 1 {
        let value = attr.pos_arg(0).unwrap();
        match value.as_constant() {
          Some(Constant::String(s)) => {
            let string_content = s.string();
            if is_valid_demand_pattern(string_content) {
              if let Some((p, l)) = self.demand_attrs.get(pred) {
                if p != string_content {
                  self.errors.push(DemandAttributeError::ConflictingPattern {
                    first_loc: l.clone(),
                    second_loc: value.location().clone(),
                  });
                }
              } else {
                let attr = (string_content.to_string(), value.location().clone());
                self.demand_attrs.insert(pred.to_string(), attr);
              }
            } else {
              self.errors.push(DemandAttributeError::InvalidPattern {
                loc: value.location().clone(),
              });
            }
          }
          Some(c) => self.errors.push(DemandAttributeError::InvalidArgumentType {
            found: c.kind().to_string(),
            loc: value.location().clone(),
          }),
          None => self.errors.push(DemandAttributeError::InvalidArgumentType {
            found: "list".to_string(),
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

impl NodeVisitor<RelationTypeDecl> for DemandAttributeAnalysis {
  fn visit(&mut self, rela_type_decl: &RelationTypeDecl) {
    for rela_type in rela_type_decl.rel_types() {
      self.process_attributes(rela_type.predicate_name(), rela_type_decl.attrs());
    }
  }
}

impl NodeVisitor<RuleDecl> for DemandAttributeAnalysis {
  fn visit(&mut self, rule_decl: &RuleDecl) {
    if rule_decl.rule().head().is_disjunction() {
      for predicate in rule_decl.rule().head().iter_predicates() {
        self.set_disjunctive(&predicate, rule_decl.rule().head().location());
        return; // early stopping because this is an error
      }
    }

    // Otherwise, we add the demand attribute
    if let Some(atom) = rule_decl.rule().head().as_atom() {
      self.process_attributes(atom.predicate().name(), rule_decl.attrs());
    }
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
    loc: NodeLocation,
  },
  InvalidArgumentType {
    found: String,
    loc: NodeLocation,
  },
  ConflictingPattern {
    first_loc: NodeLocation,
    second_loc: NodeLocation,
  },
  ArityMismatch {
    pattern: String,
    expected: usize,
    actual: usize,
    loc: NodeLocation,
  },
  InvalidPattern {
    loc: NodeLocation,
  },
  DisjunctivePredicateWithDemandAttribute {
    pred: String,
    loc: NodeLocation,
  },
}

impl FrontCompileErrorTrait for DemandAttributeError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, src: &Sources) -> String {
    match self {
      Self::InvalidNumArgs {
        pred,
        actual_num_args,
        loc,
      } => {
        format!(
          "Invalid number of arguments of @demand attribute for `{}`. Expected 1, Found {}\n{}",
          pred,
          actual_num_args,
          loc.report(src)
        )
      }
      Self::InvalidArgumentType { found, loc } => {
        format!(
          "Invalid argument type. Expected `string`, found `{}`\n{}",
          found,
          loc.report(src)
        )
      }
      Self::ConflictingPattern { first_loc, second_loc } => {
        format!(
          "Conflicting demand pattern. First defined here:\n{}re-defined here:\n{}",
          first_loc.report(src),
          second_loc.report(src)
        )
      }
      Self::ArityMismatch {
        pattern,
        expected,
        actual,
        loc,
      } => {
        format!(
          "Arity mismatch for demand pattern `{}`. Expected {}, found {}\n{}",
          pattern,
          expected,
          actual,
          loc.report(src)
        )
      }
      Self::InvalidPattern { loc } => {
        format!("Invalid demand pattern\n{}", loc.report(src))
      }
      Self::DisjunctivePredicateWithDemandAttribute { pred, loc } => {
        format!(
          "The predicate `{}` being annotated by `demand` but occurs in a disjunctive rule head\n{}",
          pred,
          loc.report(src)
        )
      }
    }
  }
}
