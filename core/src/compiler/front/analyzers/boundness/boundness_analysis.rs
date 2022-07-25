use std::collections::*;

use super::super::*;
use super::*;
use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub struct BoundnessAnalysis {
  pub rule_contexts: HashMap<Loc, (Rule, RuleContext, bool)>,
  pub errors: Vec<BoundnessAnalysisError>,
}

impl BoundnessAnalysis {
  pub fn new() -> Self {
    Self {
      rule_contexts: HashMap::new(),
      errors: Vec::new(),
    }
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
        let bounded_exprs = if let Some((pattern, _)) = demand_attrs.get(rule.head().predicate()) {
          rule
            .head()
            .iter_arguments()
            .zip(pattern.chars())
            .filter_map(|(a, b)| if b == 'b' { Some(a.clone()) } else { None })
            .collect()
        } else {
          vec![]
        };

        // Compute the boundness
        match ctx.compute_boundness(&bounded_exprs) {
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
