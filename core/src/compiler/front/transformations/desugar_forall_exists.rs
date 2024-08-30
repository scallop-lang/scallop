use crate::compiler::front::*;

/// Transforming sugar forall and exists into reduce
///
/// For example
///
/// ``` scl
/// forall(o: A(o) implies B(o))
/// ```
///
/// will be transformed into
///
/// ``` scl
/// temp_b = forall(o: A(o) implies B(o)), temp_b == true
/// ```
#[derive(Clone, Debug)]
pub struct DesugarForallExists;

impl<'a> Transformation<'a> for DesugarForallExists {}

impl DesugarForallExists {
  pub fn new() -> Self {
    Self
  }
}

impl NodeVisitor<Formula> for DesugarForallExists {
  fn visit_mut(&mut self, formula: &mut Formula) {
    match formula {
      Formula::ForallExistsReduce(r) => {
        // Get the goal
        let goal = !r.is_negated();

        // Generate a boolean variable
        let boolean_var_name = format!("r#desugar#{}", r.location_id().unwrap());
        let boolean_var_identifier = Identifier::new(boolean_var_name);
        let boolean_var = Variable::new(boolean_var_identifier);

        // Create the aggregation formula
        let reduce = Reduce::new_with_loc(
          vec![VariableOrWildcard::Variable(boolean_var.clone())],
          r.operator().clone(),
          vec![],
          r.bindings().clone(),
          r.body().clone(),
          r.group_by().clone(),
          r.location().clone(),
        );
        let reduce_formula = Formula::Reduce(reduce);

        // Create the constraint formula
        let constraint = Constraint::new(Expr::binary(BinaryExpr::new(
          BinaryOp::new_eq(),
          Expr::variable(boolean_var.clone()),
          Expr::constant(Constant::boolean(BoolLiteral::new(goal))),
        )));
        let constraint_formula = Formula::Constraint(constraint);

        // Create the conjunction of the two
        let conj = Formula::conjunction(Conjunction::new(vec![reduce_formula, constraint_formula]));

        // Update the formula
        *formula = conj;
      }
      _ => {}
    }
  }
}
