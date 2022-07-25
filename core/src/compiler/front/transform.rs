use super::transformations::*;
use super::*;

pub fn apply_transformations(ast: &mut Vec<Item>) {
  let mut transform_atomic_query = TransformAtomicQuery::new();
  let mut transform_tagged_rule = TransformTaggedRule::new();
  let mut forall_to_not_exists = TransformForall;
  let mut implies_to_disjunction = TransformImplies;
  let mut visitors = (
    &mut transform_atomic_query,
    &mut transform_tagged_rule,
    &mut forall_to_not_exists, // Note: forall needs to go before implies transformation
    &mut implies_to_disjunction,
  );
  visitors.walk_items(ast);

  // Post-transformation; annotate node ids afterwards
  let mut new_items = vec![];
  new_items.extend(transform_atomic_query.generate_items());
  new_items.extend(transform_tagged_rule.generate_items());

  // Extend the ast to incorporate these new items
  ast.extend(new_items)
}

pub trait Transformation {
  fn generate_items(self) -> Vec<Item>;
}
