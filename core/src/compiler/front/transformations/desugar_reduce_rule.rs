use crate::compiler::front::analyzers::type_inference::AggregateTypeRegistry;
use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub struct DesugarReduceRule {
  aggregate_types: AggregateTypeRegistry,
}

impl<'a> Transformation<'a> for DesugarReduceRule {}

impl DesugarReduceRule {
  pub fn new(agg_ty_registry: &AggregateTypeRegistry) -> Self {
    Self {
      aggregate_types: agg_ty_registry.clone(),
    }
  }

  fn desugar(&mut self, r: &ReduceRuleDecl) -> Option<RuleDecl> {
    let decl_loc = r.location();
    let attrs = r.attrs();
    let reduce_rule = r.rule();

    // Get the aggregate name and type
    let head_predicate = reduce_rule.head();
    let aggregate = reduce_rule.reduce();
    let aggregate_name = aggregate.operator().name().name();
    let aggregate_type = self.aggregate_types.get(aggregate_name)?;

    // Get the actual number of arg and input variables
    let num_arg = aggregate.num_args();
    let num_input = aggregate.num_bindings();

    // Solve for the expected number of output variables
    let expected_num_output = aggregate_type.infer_output_arity(num_arg, num_input).ok()?;

    // Generate a set of temporary variables
    let aggregate_id = aggregate.location_id()?;
    let vars = (0..expected_num_output)
      .map(|i| Variable::new(Identifier::new(format!("agg#{aggregate_id}#outvar#{i}"))))
      .collect::<Vec<_>>();

    // Generate a full reduce
    let generated_aggregate = Reduce::new(
      vars.iter().map(|v| VariableOrWildcard::Variable(v.clone())).collect(),
      aggregate.operator().clone(),
      aggregate.args().clone(),
      aggregate.bindings().clone(),
      aggregate.body().clone(),
      aggregate.group_by().clone(),
    );

    // Generate a head
    let generated_head_atom = Atom::new(
      head_predicate.clone(),
      vec![],
      vars.iter().map(|v| Expr::Variable(v.clone())).collect(),
    );
    let generated_head = RuleHead::Atom(generated_head_atom);

    // Generate the whole rule
    let rule = Rule::new(generated_head, Formula::Reduce(generated_aggregate));
    let rule_decl = RuleDecl::new_with_loc(attrs.clone(), None, rule, decl_loc.clone());

    // Return
    Some(rule_decl)
  }
}

impl NodeVisitor<RelationDecl> for DesugarReduceRule {
  fn visit_mut(&mut self, relation_decl: &mut RelationDecl) {
    match relation_decl {
      RelationDecl::ReduceRule(r) => {
        if let Some(f) = self.desugar(r) {
          *relation_decl = RelationDecl::Rule(f);
        }
      }
      _ => {}
    }
  }
}
