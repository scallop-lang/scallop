use scallop_core::common::foreign_aggregate::AggregateInfo;
use scallop_core::common::value_type::ValueType;
use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn test_dynamic_aggregate_count_1() {
  let ctx = unit::UnitProvenance::default();
  let rt = RuntimeEnvironment::default();

  // Relations
  let mut source_1 = DynamicRelation::<unit::UnitProvenance>::new();
  let mut source_2 = DynamicRelation::<unit::UnitProvenance>::new();
  let mut target = DynamicRelation::<unit::UnitProvenance>::new();

  // Initial
  source_1.insert_untagged(&ctx, vec![(0i8, 1i8), (1i8, 2i8), (3i8, 4i8), (3i8, 5i8)]);
  source_2.insert_untagged(&ctx, vec![(1i8, 1i8), (1i8, 2i8), (3i8, 5i8)]);

  // Iterate until fixpoint
  while source_1.changed(&ctx, rt.get_default_scheduler())
    || source_2.changed(&ctx, rt.get_default_scheduler())
    || target.changed(&ctx, rt.get_default_scheduler())
  {
    target.insert_dataflow_recent(
      &ctx,
      &DynamicDataflow::dynamic_relation(&source_1).intersect(DynamicDataflow::dynamic_relation(&source_2), &ctx),
      &rt,
    )
  }

  let completed_target = target.complete(&ctx);

  let mut first_time = true;
  let mut agg = DynamicRelation::<unit::UnitProvenance>::new();
  while agg.changed(&ctx, rt.get_default_scheduler()) || first_time {
    agg.insert_dataflow_recent(
      &ctx,
      &DynamicDataflow::new(DynamicAggregationSingleGroupDataflow::new(
        rt.aggregate_registry
          .instantiate_aggregator(
            "count",
            AggregateInfo::default().with_input_var_types(vec![ValueType::I8, ValueType::I8]),
          )
          .unwrap(),
        DynamicDataflow::dynamic_collection(&completed_target, first_time),
        &ctx,
        &rt,
      )),
      &rt,
    );
    first_time = false;
  }

  expect_collection(&agg.complete(&ctx), vec![2usize]);
}
