use super::super::*;

pub fn remove_false_rules(rules: &mut Vec<Rule>) {
  rules.retain(|rule| {
    rule
      .body_literals()
      .find(|l| match l {
        Literal::False => true,
        _ => false,
      })
      .is_none()
  })
}
