use scallop_core::common::aggregate_op::AggregateOp;
use scallop_core::common::expr::*;
use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn test_dynamic_group_and_count_1() {
  let mut ctx = unit::UnitProvenance;
  let mut rt = RuntimeEnvironment::new();

  // Relations
  let mut color = DynamicRelation::<unit::UnitProvenance>::new();
  let mut rev_color = DynamicRelation::<unit::UnitProvenance>::new();

  // Initial
  color.insert_untagged(
    &mut ctx,
    vec![
      (0usize, "red"),
      (1usize, "red"),
      (2usize, "green"),
      (3usize, "green"),
      (4usize, "green"),
      (5usize, "blue"),
    ],
  );

  // Iterate until fixpoint
  while color.changed(&ctx) || rev_color.changed(&ctx) {
    rev_color.insert_dataflow_recent(
      &ctx,
      &DynamicDataflow::project(
        DynamicDataflow::dynamic_relation(&color),
        (Expr::access(1), Expr::access(0)).into(),
      ),
      &mut rt,
    )
  }

  // Complete rev_color
  let completed_rev_color = rev_color.complete(&ctx);

  // Group and aggregate
  let mut first_time = true;
  let mut color_count = DynamicRelation::<unit::UnitProvenance>::new();
  while color_count.changed(&ctx) || first_time {
    color_count.insert_dataflow_recent(
      &ctx,
      &DynamicAggregationDataflow::implicit(
        AggregateOp::Count.into(),
        DynamicDataflow::dynamic_collection(&completed_rev_color, first_time),
        &ctx,
      )
      .into(),
      &mut rt,
    );
    first_time = false;
  }

  expect_collection(
    &color_count.complete(&ctx),
    vec![("red", 2usize), ("green", 3usize), ("blue", 1usize)],
  );
}

#[test]
fn test_dynamic_group_count_max_1() {
  let mut ctx = unit::UnitProvenance;
  let mut rt = RuntimeEnvironment::default();

  // Relations
  let mut color = DynamicRelation::<unit::UnitProvenance>::new();
  let mut rev_color = DynamicRelation::<unit::UnitProvenance>::new();

  // Initial
  color.insert_untagged(
    &mut ctx,
    vec![
      (0usize, "red"),
      (1usize, "red"),
      (2usize, "green"),
      (3usize, "green"),
      (4usize, "green"),
      (5usize, "blue"),
    ],
  );

  // Iterate until fixpoint
  while color.changed(&ctx) || rev_color.changed(&ctx) {
    rev_color.insert_dataflow_recent(
      &ctx,
      &DynamicDataflow::project(
        DynamicDataflow::dynamic_relation(&color),
        Expr::Tuple(vec![Expr::Access(1.into()), Expr::Access(0.into())]),
      ),
      &mut rt,
    )
  }

  // Complete rev_color
  let completed_rev_color = rev_color.complete(&ctx);

  // Group and aggregate
  let mut iter_1_first_time = true;
  let mut color_count = DynamicRelation::<unit::UnitProvenance>::new();
  while color_count.changed(&ctx) || iter_1_first_time {
    color_count.insert_dataflow_recent(
      &ctx,
      &DynamicAggregationDataflow::implicit(
        AggregateOp::Count.into(),
        DynamicDataflow::dynamic_collection(&completed_rev_color, iter_1_first_time),
        &ctx,
      )
      .into(),
      &mut rt,
    );
    iter_1_first_time = false;
  }

  // Complete agg
  let completed_color_count = color_count.complete(&ctx);

  // Find Max
  let mut iter_2_first_time = true;
  let mut max_count_color = DynamicRelation::<unit::UnitProvenance>::new();
  while max_count_color.changed(&ctx) || iter_2_first_time {
    max_count_color.insert_dataflow_recent(
      &ctx,
      &DynamicAggregationDataflow::single(
        AggregateOp::Argmax.into(),
        DynamicDataflow::dynamic_collection(&completed_color_count, iter_2_first_time),
        &ctx,
      )
      .into(),
      &mut rt,
    );
    iter_2_first_time = false;
  }

  expect_collection(&max_count_color.complete(&ctx), vec![("green", 3usize)]);
}
