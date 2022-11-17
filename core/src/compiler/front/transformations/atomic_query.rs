use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub struct TransformAtomicQuery {
  pub to_add_rules: Vec<Rule>,
}

impl TransformAtomicQuery {
  pub fn new() -> Self {
    Self { to_add_rules: vec![] }
  }
}

impl NodeVisitorMut for TransformAtomicQuery {
  fn visit_query(&mut self, query: &mut Query) {
    match &query.node {
      QueryNode::Atom(a) => {
        let query_name = format!("{}", a);
        let args = a
          .iter_arguments()
          .enumerate()
          .map(|(i, v)| {
            if v.is_variable() || v.is_constant() {
              v.clone()
            } else {
              let name = format!("qa#{}", i);
              let id = IdentifierNode::new(name);
              let var = VariableNode::new(id.into());
              Expr::Variable(var.into())
            }
          })
          .collect::<Vec<_>>();
        let head_atom = AtomNode {
          predicate: IdentifierNode::new(query_name.clone()).into(),
          args: args.clone(),
        };
        let body_atom = AtomNode {
          predicate: IdentifierNode::new(a.predicate().clone()).into(),
          args: args.clone(),
        };
        let eq_constraints = a
          .iter_arguments()
          .enumerate()
          .filter_map(|(i, arg)| {
            if arg.is_wildcard() || arg.is_variable() || arg.is_constant() {
              None
            } else {
              let bin_expr = Expr::binary(BinaryOpNode::Eq.into(), args[i].clone(), arg.clone());
              let constraint = Formula::Constraint(ConstraintNode { expr: bin_expr }.into());
              Some(constraint)
            }
          })
          .collect();
        let conj = ConjunctionNode {
          args: vec![vec![Formula::Atom(body_atom.into())], eq_constraints].concat(),
        };
        let rule = RuleNode {
          head: head_atom.into(),
          body: Formula::Conjunction(conj.into()),
        };
        self.to_add_rules.push(rule.into());

        // Transform this query into a predicate query
        query.node = QueryNode::Predicate(IdentifierNode::new(query_name).into());
      }
      _ => {}
    }
  }
}

impl Transformation for TransformAtomicQuery {
  fn generate_items(self) -> Vec<Item> {
    self
      .to_add_rules
      .iter()
      .map(|rule| {
        let rule_decl = RuleDeclNode {
          attrs: vec![],
          tag: Tag::default_none(),
          rule: rule.clone(),
        };
        let rel_decl = RelationDeclNode::Rule(rule_decl.into());
        let item = Item::RelationDecl(rel_decl.into());
        item
      })
      .collect()
  }
}
