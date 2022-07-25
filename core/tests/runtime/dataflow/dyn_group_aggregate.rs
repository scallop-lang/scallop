use scallop_core::common::expr::*;
use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn test_dynamic_group_and_count_1() {
  let mut ctx = unit::UnitContext;

  // Relations
  let mut color = DynamicRelation::<unit::Unit>::new();
  let mut rev_color = DynamicRelation::<unit::Unit>::new();

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
    )
  }

  // Complete rev_color
  let completed_rev_color = rev_color.complete(&ctx);

  // Group and aggregate
  let mut first_time = true;
  let mut color_count = DynamicRelation::<unit::Unit>::new();
  while color_count.changed(&ctx) || first_time {
    first_time = false;

    color_count.insert_dataflow_recent(
      &ctx,
      &DynamicGroups::group_from_collection(&completed_rev_color)
        .aggregate(DynamicAggregateOp::count(Expr::access(())), &ctx),
    );
  }

  expect_collection(
    &color_count.complete(&ctx),
    vec![("red", 2usize), ("green", 3usize), ("blue", 1usize)],
  );
}

#[test]
fn test_dynamic_group_count_max_1() {
  let mut ctx = unit::UnitContext;

  // Relations
  let mut color = DynamicRelation::<unit::Unit>::new();
  let mut rev_color = DynamicRelation::<unit::Unit>::new();

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
    )
  }

  // Complete rev_color
  let completed_rev_color = rev_color.complete(&ctx);

  // Group and aggregate
  let mut iter_1_first_time = true;
  let mut color_count = DynamicRelation::<unit::Unit>::new();
  while color_count.changed(&ctx) || iter_1_first_time {
    iter_1_first_time = false;

    color_count.insert_dataflow_recent(
      &ctx,
      &DynamicGroups::group_from_collection(&completed_rev_color)
        .aggregate(DynamicAggregateOp::count(Expr::access(())), &ctx),
    );
  }

  // Complete agg
  let completed_color_count = color_count.complete(&ctx);

  // Find Max
  let mut iter_2_first_time = true;
  let mut max_count_color = DynamicRelation::<unit::Unit>::new();
  while max_count_color.changed(&ctx) || iter_2_first_time {
    iter_2_first_time = false;

    max_count_color.insert_dataflow_recent(
      &ctx,
      &DynamicGroups::SingleCollection(&completed_color_count).aggregate(
        DynamicAggregateOp::max(Some(Expr::access(0)), Expr::access(1)),
        &ctx,
      ),
    )
  }

  expect_collection(&max_count_color.complete(&ctx), vec![("green", 3usize)]);
}
