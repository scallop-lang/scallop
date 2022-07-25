use scallop_core::common::expr::*;
use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn test_dynamic_aggregate_count_1() {
  let mut ctx = unit::UnitContext::default();

  // Relations
  let mut source_1 = DynamicRelation::<unit::Unit>::new();
  let mut source_2 = DynamicRelation::<unit::Unit>::new();
  let mut target = DynamicRelation::<unit::Unit>::new();

  // Initial
  source_1.insert_untagged(
    &mut ctx,
    vec![(0i8, 1i8), (1i8, 2i8), (3i8, 4i8), (3i8, 5i8)],
  );
  source_2.insert_untagged(&mut ctx, vec![(1i8, 1i8), (1i8, 2i8), (3i8, 5i8)]);

  // Iterate until fixpoint
  while source_1.changed(&ctx) || source_2.changed(&ctx) || target.changed(&ctx) {
    target.insert_dataflow_recent(
      &ctx,
      &DynamicDataflow::from(&source_1).intersect(DynamicDataflow::from(&source_2), &ctx),
    )
  }

  let completed_target = target.complete(&ctx);

  let mut first_time = true;
  let mut agg = DynamicRelation::<unit::Unit>::new();
  while agg.changed(&ctx) || first_time {
    first_time = false;
    agg.insert_dataflow_recent(
      &ctx,
      &DynamicGroups::from_collection(&completed_target)
        .aggregate(DynamicAggregateOp::count(Expr::access(())), &ctx),
    );
  }

  expect_collection(&agg.complete(&ctx), vec![2usize]);
}
