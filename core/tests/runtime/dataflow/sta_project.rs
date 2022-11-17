use scallop_core::runtime::provenance::*;
use scallop_core::runtime::statics::dataflow::*;
use scallop_core::runtime::statics::*;
use scallop_core::testing::*;

#[test]
fn test_static_project_1() {
  let mut ctx = unit::UnitProvenance;

  // Relations
  let mut source = StaticRelation::<(i8, i8), unit::UnitProvenance>::new();
  let mut target = StaticRelation::<(i8, i8), unit::UnitProvenance>::new();

  // Initial
  source.insert_untagged(&mut ctx, vec![(0, 1), (1, 2)]);

  // Iterate until fixpoint
  while source.changed(&ctx) || target.changed(&ctx) {
    target.insert_dataflow_recent(&ctx, project(&source, |t| (t.0, t.1 + 1)), true)
  }

  expect_static_collection(&target.complete(&ctx), vec![(0, 2), (1, 3)]);
}
