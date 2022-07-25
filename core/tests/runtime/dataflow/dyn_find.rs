use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn test_dynamic_find_1() {
  let mut ctx = unit::UnitContext;

  // Relations
  let mut source = DynamicRelation::<unit::Unit>::new();
  let mut target = DynamicRelation::<unit::Unit>::new();

  // Initial
  source.insert_untagged(&mut ctx, vec![(0i8, 1i8), (1i8, 2i8)]);

  // Iterate until fixpoint
  while source.changed(&ctx) || target.changed(&ctx) {
    target.insert_dataflow_recent(
      &ctx,
      &DynamicDataflow::find(DynamicDataflow::dynamic_relation(&source), 1i8.into()),
    )
  }

  expect_collection(&source.complete(&ctx), vec![(0i8, 1i8), (1i8, 2i8)]);
  expect_collection(&target.complete(&ctx), vec![(1i8, 2i8)]);
}
