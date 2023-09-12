use scallop_core::common::expr::*;
use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn test_dynamic_filter_1() {
  let mut ctx = unit::UnitProvenance;
  let rt = RuntimeEnvironment::default();

  // Relations
  let mut source = DynamicRelation::<unit::UnitProvenance>::new();
  let mut target = DynamicRelation::<unit::UnitProvenance>::new();

  // Initial
  source.insert_untagged(&mut ctx, vec![(0i8, 1i8), (1i8, 2i8)]);

  // Iterate until fixpoint
  while source.changed(&ctx) || target.changed(&ctx) {
    target.insert_dataflow_recent(
      &ctx,
      &DynamicDataflow::filter(
        DynamicDataflow::dynamic_relation(&source),
        Expr::access(1).gt(Expr::constant(1i8)),
        &rt,
      ),
      &rt,
    )
  }

  expect_collection(&source.complete(&ctx), vec![(0i8, 1i8), (1i8, 2i8)]);
  expect_collection(&target.complete(&ctx), vec![(1i8, 2i8)]);
}
