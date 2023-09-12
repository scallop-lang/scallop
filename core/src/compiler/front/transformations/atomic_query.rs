use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub struct TransformAtomicQuery {
  pub to_add_rules: Vec<Rule>,
}

impl TransformAtomicQuery {
  pub fn new() -> Self {
    Self { to_add_rules: vec![] }
  }

  pub fn drain_items(self) -> Vec<Item> {
    self
      .to_add_rules
      .iter()
      .map(|rule| {
        let rule_decl = RuleDecl::new(vec![], Tag::none(), rule.clone());
        let rel_decl = RelationDecl::Rule(rule_decl);
        let item = Item::RelationDecl(rel_decl);
        item
      })
      .collect()
  }
}

impl NodeVisitor<Query> for TransformAtomicQuery {
  fn visit_mut(&mut self, query: &mut Query) {
    match query {
      Query::Atom(a) => {
        let query_name = format!("{}", a);
        let args = a
          .iter_args()
          .enumerate()
          .map(|(i, v)| {
            if v.is_variable() || v.is_constant() {
              v.clone()
            } else {
              let name = format!("qa#{}", i);
              Expr::Variable(Variable::new(Identifier::new(name)))
            }
          })
          .collect::<Vec<_>>();
        let head_atom = Atom::new(Identifier::new(query_name.clone()), Vec::new(), args.clone());
        let body_atom = Atom::new(Identifier::new(a.predicate().name().clone()), Vec::new(), args.clone());
        let eq_constraints = a
          .iter_args()
          .enumerate()
          .filter_map(|(i, arg)| {
            if arg.is_wildcard() || arg.is_variable() || arg.is_constant() {
              None
            } else {
              let bin_expr = Expr::binary(BinaryExpr::new(BinaryOp::new_eq(), args[i].clone(), arg.clone()));
              let constraint = Formula::Constraint(Constraint::new(bin_expr));
              Some(constraint)
            }
          })
          .collect();
        let conj = Conjunction::new(vec![vec![Formula::Atom(body_atom.into())], eq_constraints].concat());
        let rule = Rule::new(RuleHead::atom(head_atom), Formula::Conjunction(conj));
        self.to_add_rules.push(rule);

        // Transform this query into a predicate query
        *query = Query::Predicate(Identifier::new(query_name));
      }
      _ => {}
    }
  }
}
