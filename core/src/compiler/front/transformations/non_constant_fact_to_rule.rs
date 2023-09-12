use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub struct TransformNonConstantFactToRule;

impl NodeVisitor<RelationDecl> for TransformNonConstantFactToRule {
  fn visit_mut(&mut self, relation_decl: &mut RelationDecl) {
    // First collect the expressions in the fact that is not constant
    let (attrs, tag, head, non_const_var_expr_pairs) = match &relation_decl {
      RelationDecl::Fact(f) => {
        let attrs = f.attrs().clone();
        let tag = f.tag().clone();
        let head = f.atom().clone();
        let non_const = head
          .iter_args()
          .enumerate()
          .filter_map(|(i, e)| if e.is_constant() { None } else { Some((i, e.clone())) })
          .collect::<Vec<_>>();
        (attrs, tag, head, non_const)
      }
      _ => return,
    };

    // Check if there is non-constant
    if non_const_var_expr_pairs.is_empty() {
      return;
    }

    // Transform this into a rule. First generate the head atom:
    // all the non-constant arguments will be replaced by a variable
    let head_atom = Atom::new(
      head.predicate().clone(),
      vec![],
      head
        .iter_args()
        .enumerate()
        .map(|(i, e)| {
          if e.is_constant() {
            e.clone()
          } else {
            let var = Variable::new(Identifier::new(format!("fnc#{}", i)));
            Expr::Variable(var.into())
          }
        })
        .collect(),
    );
    let head: RuleHead = head_atom.into();

    // For each non-constant variable, we create a equality constraint
    let eq_consts = non_const_var_expr_pairs
      .into_iter()
      .map(|(i, e)| {
        let var_expr = Expr::variable(Variable::new(Identifier::new(format!("fnc#{}", i))));
        let eq_expr = Expr::binary(BinaryExpr::new(BinaryOp::new_eq(), var_expr, e));
        Formula::Constraint(Constraint::new(eq_expr))
      })
      .collect::<Vec<_>>();
    let body = Formula::conjunction(Conjunction::new(eq_consts));

    // Finally, generate a rule declaration
    let rule = Rule::new(head, body);
    let rule_decl = RuleDecl::new(attrs, tag, rule);

    // Modify the original relation declaration
    *relation_decl = RelationDecl::rule(rule_decl);
  }
}
