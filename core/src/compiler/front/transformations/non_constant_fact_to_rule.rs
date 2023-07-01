use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub struct TransformNonConstantFactToRule;

impl NodeVisitorMut for TransformNonConstantFactToRule {
  fn visit_relation_decl(&mut self, relation_decl: &mut RelationDecl) {
    // First collect the expressions in the fact that is not constant
    let (attrs, tag, head, non_const_var_expr_pairs) = match &relation_decl.node {
      RelationDeclNode::Fact(f) => {
        let attrs = f.node.attrs.clone();
        let tag = f.node.tag.clone();
        let head = f.atom().clone();
        let non_const = head
          .iter_arguments()
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
    let head_atom: Atom = AtomNode {
      predicate: head.node.predicate.clone(),
      type_args: vec![],
      args: head
        .iter_arguments()
        .enumerate()
        .map(|(i, e)| {
          if e.is_constant() {
            e.clone()
          } else {
            let id = IdentifierNode::new(format!("fnc#{}", i));
            let var = VariableNode::new(id.into());
            Expr::Variable(var.into())
          }
        })
        .collect(),
    }
    .into();
    let head: RuleHead = head_atom.into();

    // For each non-constant variable, we create a equality constraint
    let eq_consts = non_const_var_expr_pairs
      .into_iter()
      .map(|(i, e)| {
        let id = IdentifierNode::new(format!("fnc#{}", i));
        let var = VariableNode::new(id.into());
        let var_expr = Expr::Variable(var.into());
        let eq_expr = Expr::binary(BinaryOp::default_eq(), var_expr, e);
        Formula::Constraint(ConstraintNode::new(eq_expr).into())
      })
      .collect::<Vec<_>>();
    let body = Formula::conjunction(eq_consts);

    // Finally, generate a rule declaration
    let rule = RuleNode::new(head, body);
    let rule_decl = RuleDeclNode::new(attrs, tag, rule.into());

    // Modify the original relation declaration
    *relation_decl = RelationDecl::new(
      relation_decl.location().clone(),
      RelationDeclNode::Rule(rule_decl.into()),
    );
  }
}
