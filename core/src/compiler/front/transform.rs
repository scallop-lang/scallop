use super::transformations::*;
use super::*;

pub fn apply_transformations(ast: &mut Vec<Item>, analysis: &mut Analysis) {
  let mut transform_adt = TransformAlgebraicDataType::new(&mut analysis.adt_analysis);
  let mut transform_const_var_to_const = TransformConstVarToConst::new(&analysis.constant_decl_analysis);
  let mut transform_atomic_query = TransformAtomicQuery::new();
  let mut transform_conjunctive_head = TransformConjunctiveHead::new();
  let mut transform_tagged_rule = TransformTaggedRule::new();
  let mut transform_non_const_fact = TransformNonConstantFactToRule;
  let mut desugar_arg_type_adornment = DesugarArgTypeAdornment::new();
  let mut desugar_case_is = DesugarCaseIs::new();
  let mut desugar_forall_exists = DesugarForallExists::new();
  let mut desugar_range = DesugarRange::new();
  let mut forall_to_not_exists = TransformForall;
  let mut implies_to_disjunction = TransformImplies;
  let mut visitors = (
    &mut transform_adt,
    &mut transform_atomic_query,
    &mut transform_conjunctive_head,
    &mut transform_const_var_to_const,
    &mut transform_tagged_rule,
    &mut transform_non_const_fact,
    &mut desugar_arg_type_adornment,
    &mut desugar_case_is,
    &mut desugar_forall_exists,
    &mut desugar_range,
    &mut forall_to_not_exists, // Note: forall needs to go before implies transformation
    &mut implies_to_disjunction,
  );
  ast.walk_mut(&mut visitors);

  // Post-transformation; remove items
  ast.retain(|item| transform_conjunctive_head.retain(item) && transform_adt.retain(item));

  // Post-transformation; annotate node ids afterwards
  let mut new_items = vec![];
  new_items.extend(transform_adt.generate_items());
  new_items.extend(transform_const_var_to_const.generate_items());
  new_items.extend(transform_atomic_query.drain_items());
  new_items.extend(transform_conjunctive_head.generate_items());
  new_items.extend(transform_tagged_rule.drain_items());

  // Some of the transformations need to be applied to new items as well
  new_items.walk_mut(&mut transform_const_var_to_const);

  // Extend the ast to incorporate these new items
  ast.extend(new_items)
}
