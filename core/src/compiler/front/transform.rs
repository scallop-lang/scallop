use super::transformations::*;
use super::*;

pub fn apply_transformations(ast: &mut Vec<Item>, analysis: &Analysis) {
  let mut transform_const_var_to_const = TransformConstVarToConst::new(&analysis.constant_decl_analysis);
  let mut transform_atomic_query = TransformAtomicQuery::new();
  let mut transform_tagged_rule = TransformTaggedRule::new();
  let mut transform_non_const_fact = TransformNonConstantFactToRule;
  let mut desugar_forall_exists = DesugarForallExists::new();
  let mut forall_to_not_exists = TransformForall;
  let mut implies_to_disjunction = TransformImplies;
  let mut visitors = (
    &mut transform_atomic_query,
    &mut transform_const_var_to_const,
    &mut transform_tagged_rule,
    &mut transform_non_const_fact,
    &mut desugar_forall_exists,
    &mut forall_to_not_exists, // Note: forall needs to go before implies transformation
    &mut implies_to_disjunction,
  );
  visitors.walk_items(ast);

  // Post-transformation; annotate node ids afterwards
  let mut new_items = vec![];
  new_items.extend(transform_atomic_query.generate_items());
  new_items.extend(transform_tagged_rule.generate_items());

  // Some of the transformations need to be applied to new items as well
  transform_const_var_to_const.walk_items(&mut new_items);

  // Extend the ast to incorporate these new items
  ast.extend(new_items)
}

pub trait Transformation {
  fn generate_items(self) -> Vec<Item>;
}
