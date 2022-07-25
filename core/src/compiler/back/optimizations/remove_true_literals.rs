use super::super::*;

pub fn remove_true_literals(rule: &mut Rule) {
  rule.body.args.retain(|l| match l {
    Literal::True => false,
    _ => true,
  })
}
