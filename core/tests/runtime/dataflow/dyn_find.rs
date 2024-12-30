use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn test_dynamic_find_1() {
  let mut ctx = unit::UnitProvenance;
  let mut rt = RuntimeEnvironment::default();

  // Relations
  let mut source = DynamicRelation::<unit::UnitProvenance>::new();
  let mut target = DynamicRelation::<unit::UnitProvenance>::new();

  // Initial
  source.insert_untagged(&mut ctx, vec![(0i8, 1i8), (1i8, 2i8)]);

  // Iterate until fixpoint
  while source.changed(&ctx, rt.get_default_scheduler()) || target.changed(&ctx, rt.get_default_scheduler()) {
    target.insert_dataflow_recent(
      &ctx,
      &DynamicDataflow::find(DynamicDataflow::dynamic_relation(&source), 1i8.into()),
      &mut rt,
    )
  }

  expect_collection(&source.complete(&ctx).into(), vec![(0i8, 1i8), (1i8, 2i8)]);
  expect_collection(&target.complete(&ctx).into(), vec![(1i8, 2i8)]);
}
