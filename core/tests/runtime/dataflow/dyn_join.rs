use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn test_dynamic_join_1() {
  let mut ctx = unit::UnitContext;

  // Relations
  let mut source_1 = DynamicRelation::<unit::Unit>::new();
  let mut source_2 = DynamicRelation::<unit::Unit>::new();
  let mut target = DynamicRelation::<unit::Unit>::new();

  // Initial
  source_1.insert_untagged(&mut ctx, vec![(0i8, 1i8), (1i8, 2i8)]);
  source_2.insert_untagged(&mut ctx, vec![(0i8, 2i8), (1i8, 5i8)]);

  // Iterate until fixpoint
  while source_1.changed(&ctx) || source_2.changed(&ctx) || target.changed(&ctx) {
    target.insert_dataflow_recent(
      &ctx,
      &DynamicDataflow::dynamic_relation(&source_1).join(DynamicDataflow::dynamic_relation(&source_2), &ctx),
    )
  }

  expect_collection(&target.complete(&ctx), vec![(0i8, 1i8, 2i8), (1i8, 2i8, 5i8)]);
}
