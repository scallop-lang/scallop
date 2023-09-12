use crate::compiler::front::*;

/// Transforming sugar range formula with range atom
///
/// For example
///
/// ``` scl
/// rel grid(x, y) = x in 0..10 and y in 3..=5
/// ```
///
/// will be transformed into
///
/// ``` scl
/// rel grid(x, y) = range<i32>(0, 10, x) and range<i32>(3, 6, y)
/// ```
#[derive(Clone, Debug)]
pub struct DesugarRange;

impl DesugarRange {
  pub fn new() -> Self {
    Self
  }
}

impl NodeVisitor<Formula> for DesugarRange {
  fn visit_mut(&mut self, formula: &mut Formula) {
    match formula {
      Formula::Range(r) => {
        let range_atom = Atom::new_with_loc(
          Identifier::new("range".to_string()),
          vec![Type::i32()],
          match r.inclusive() {
            true => vec![
              r.lower().clone(),
              Expr::binary(
                BinaryExpr::new(
                  BinaryOp::new_add(),
                  r.upper().clone(),
                  Expr::Constant(Constant::integer(IntLiteral::new(1))),
                )
              ),
              Expr::Variable(r.left().clone()),
            ],
            false => vec![
              r.lower().clone(),
              r.upper().clone(),
              Expr::Variable(r.left().clone()),
            ],
          },
          r.location().clone(),
        );

        // Update the formula
        *formula = Formula::Atom(range_atom);
      }
      _ => {}
    }
  }
}
