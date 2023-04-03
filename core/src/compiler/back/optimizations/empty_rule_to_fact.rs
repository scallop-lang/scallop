use crate::common::input_tag::*;

use super::super::*;

pub fn empty_rule_to_fact(rules: &mut Vec<Rule>, facts: &mut Vec<Fact>) {
  rules.retain(|rule| {
    if rule.body.args.is_empty() {
      // Create fact
      let fact = Fact {
        tag: DynamicInputTag::None,
        predicate: rule.head.predicate.clone(),
        args: rule
          .head
          .args
          .iter()
          .map(|arg| match arg {
            Term::Constant(c) => c.clone(),
            Term::Variable(v) => panic!("[Internal Error] Invalid head variable `{}` in an empty rule", v.name),
          })
          .collect(),
      };

      // Add the fact to the set of facts
      facts.push(fact);

      false
    } else {
      true
    }
  })
}
