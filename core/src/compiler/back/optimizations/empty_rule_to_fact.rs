use crate::common::input_tag::*;

use super::super::*;

pub fn empty_rule_to_fact(rules: &mut Vec<Rule>, facts: &mut Vec<Fact>) {
  rules.retain(|rule| {
    if rule.body.args.is_empty() {
      match &rule.head {
        Head::Atom(head_atom) => {
          // Create fact
          let fact = Fact {
            tag: DynamicInputTag::None,
            predicate: head_atom.predicate.clone(),
            args: head_atom
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
        }
        Head::Disjunction(_) => {
          // TODO: Handle disjunctions
          true
        }
      }
    } else {
      true
    }
  })
}
