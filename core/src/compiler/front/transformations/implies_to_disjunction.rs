use crate::compiler::front::*;

/// Transforming implies into disjunction, e.g. `A -> B` should be rewritten to `(not A) or B`
#[derive(Clone, Debug)]
pub struct TransformImplies;

impl NodeVisitorMut for TransformImplies {
  fn visit_formula(&mut self, formula: &mut Formula) {
    match formula {
      Formula::Implies(i) => {
        let rewrite = Formula::Disjunction(Disjunction::new(
          i.location().clone(),
          DisjunctionNode {
            args: vec![i.left().negate(), i.right().clone()],
          },
        ));
        *formula = rewrite;
      }
      _ => {}
    }
  }
}
