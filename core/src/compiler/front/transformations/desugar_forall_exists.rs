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

impl DesugarForallExists {
  pub fn new() -> Self {
    Self
  }
}

impl NodeVisitorMut for DesugarForallExists {
  fn visit_formula(&mut self, formula: &mut Formula) {
    match formula {
      Formula::ForallExistsReduce(r) => {
        // Generate a boolean variable
        let boolean_var_name = format!("r#desugar#{}", r.loc.id.unwrap());
        let boolean_var_identifier: Identifier = IdentifierNode::new(boolean_var_name).into();
        let boolean_var: Variable = VariableNode::new(boolean_var_identifier).into();

        // Create the aggregation formula
        let reduce = Reduce {
          node: ReduceNode {
            operator: r.node.operator.clone(),
            left: vec![VariableOrWildcard::Variable(boolean_var.clone())],
            args: vec![],
            bindings: r.node.bindings.clone(),
            body: r.node.body.clone(),
            group_by: r.node.group_by.clone(),
          },
          loc: r.loc.clone(),
        };
        let reduce_formula = Formula::Reduce(reduce);

        // Create the constraint formula
        let constraint = Constraint::default_with_expr(Expr::binary(
          BinaryOp::default_eq(),
          Expr::Variable(boolean_var.clone()),
          Expr::boolean_true(),
        ));
        let constraint_formula = Formula::Constraint(constraint);

        // Create the conjunction of the two
        let conj = Formula::conjunction(vec![reduce_formula, constraint_formula]);

        // Update the formula
        *formula = conj;
      }
      _ => {}
    }
  }
}
