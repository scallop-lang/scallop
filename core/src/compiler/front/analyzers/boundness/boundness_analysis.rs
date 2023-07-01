use std::collections::*;

use super::super::*;
use super::*;

use crate::common::foreign_predicate::*;
use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub struct BoundnessAnalysis {
  pub predicate_bindings: ForeignPredicateBindings,
  pub rule_contexts: HashMap<Loc, (Rule, RuleContext, bool)>,
  pub errors: Vec<BoundnessAnalysisError>,
}

impl BoundnessAnalysis {
  pub fn new(registry: &ForeignPredicateRegistry) -> Self {
    Self {
      predicate_bindings: registry.into(),
      rule_contexts: HashMap::new(),
      errors: Vec::new(),
    }
  }

  pub fn add_foreign_predicate<F: ForeignPredicate>(&mut self, fp: &F) {
    self.predicate_bindings.add(fp)
  }

  pub fn get_rule_context(&self, loc: &Loc) -> Option<&RuleContext> {
    self.rule_contexts.get(loc).map(|(_, ctx, _)| ctx)
  }

  pub fn check_boundness(&mut self, demand_attr_analysis: &DemandAttributeAnalysis) {
    let demand_attrs = &demand_attr_analysis.demand_attrs;

    // For each rule context, do boundness check
    for (_, (rule, ctx, inferred)) in &mut self.rule_contexts {
      if !*inferred {
        // Make sure the demand attribute is affecting boundness analysis,
        // through some of the head expressions being bounded
        let bounded_exprs = if let Some(head_atom) = rule.head().atom() {
          if let Some((pattern, _)) = demand_attrs.get(&head_atom.predicate()) {
            head_atom
              .iter_arguments()
              .zip(pattern.chars())
              .filter_map(|(a, b)| if b == 'b' { Some(a.clone()) } else { None })
              .collect()
          } else {
            vec![]
          }
        } else {
          vec![]
        };

        // Compute the boundness
        match ctx.compute_boundness(&self.predicate_bindings, &bounded_exprs) {
          Ok(_) => {}
          Err(errs) => {
            self.errors.extend(errs);
          }
        }

        // Set inferred to true
        *inferred = true;
      }
    }
  }
}

impl NodeVisitor for BoundnessAnalysis {
  fn visit_rule(&mut self, rule: &Rule) {
    let ctx = RuleContext::from_rule(rule);
    self
      .rule_contexts
      .insert(rule.location().clone(), (rule.clone(), ctx, false));
  }
}
