use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn test_dynamic_difference_unit_1() {
  test_dynamic_difference_master_1(unit::UnitContext::default());
}

#[test]
fn test_dynamic_difference_bool_1() {
  test_dynamic_difference_master_1(boolean::BooleanContext::default());
}

fn test_dynamic_difference_master_1<C>(mut ctx: C)
where
  C::Tag: std::fmt::Debug,
  C: ProvenanceContext,
{
  // Relations
  let mut source_1 = DynamicRelation::<C::Tag>::new();
  let mut source_2 = DynamicRelation::<C::Tag>::new();
  let mut target = DynamicRelation::<C::Tag>::new();

  // Initial
  source_1.insert_untagged(&mut ctx, vec![(0i8, 1i8), (1i8, 2i8)]);
  source_2.insert_untagged(&mut ctx, vec![(1i8, 1i8), (1i8, 2i8)]);

  // To allow source_2 for computation, we need it to be a collection
  while source_2.changed(&ctx) {}
  let source_2_coll = source_2.complete(&ctx);

  // Iterate until fixpoint
  while source_1.changed(&ctx) || target.changed(&ctx) {
    target.insert_dataflow_recent(
      &ctx,
      &DynamicDataflow::from(&source_1).difference(
        DynamicDataflow::dynamic_recent_collection(&source_2_coll),
        &ctx,
      ),
    )
  }

  let result = target.complete(&ctx);
  expect_collection(&result, vec![(0i8, 1i8)]);
}
