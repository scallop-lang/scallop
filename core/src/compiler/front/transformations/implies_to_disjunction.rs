use crate::compiler::front::*;

/// Transforming implies into disjunction, e.g. `A -> B` should be rewritten to `(not A) or B`
#[derive(Clone, Debug)]
pub struct TransformImplies;

impl NodeVisitor<Formula> for TransformImplies {
  fn visit_mut(&mut self, formula: &mut Formula) {
    match formula {
      Formula::Implies(i) => {
        let rewrite = Formula::Disjunction(Disjunction::new_with_loc(
          vec![i.left().negate(), i.right().clone()],
          i.location().clone(),
        ));
        *formula = rewrite;
      }
      _ => {}
    }
  }
}
