use scallop_core::common::expr::*;
use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn test_dyn_project_1() {
  let mut ctx = unit::UnitProvenance;
  let rt = RuntimeEnvironment::new_std();

  // Relations
  let mut source = DynamicRelation::<unit::UnitProvenance>::new();
  let mut target = DynamicRelation::<unit::UnitProvenance>::new();

  // Initial
  source.insert_untagged(&mut ctx, vec![(0i8, 1i8), (1i8, 2i8)]);

  // Iterate until fixpoint
  while source.changed(&ctx) || target.changed(&ctx) {
    target.insert_dataflow_recent(
      &ctx,
      &DynamicDataflow::project(
        DynamicDataflow::dynamic_relation(&source),
        (Expr::access(0), Expr::access(1) + Expr::constant(1i8)).into(),
        &rt,
      ),
      &rt,
    )
  }

  expect_collection(&target.complete(&ctx), vec![(0i8, 2i8), (1i8, 3i8)]);
}
