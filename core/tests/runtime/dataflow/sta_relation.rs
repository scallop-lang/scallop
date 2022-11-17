use scallop_core::runtime::provenance::*;
use scallop_core::runtime::statics::*;
use scallop_core::testing::*;

#[test]
fn simple_static_relation_dataflow() {
  let mut ctx = unit::UnitProvenance;

  // Relations
  let mut source = StaticRelation::<(usize, usize), unit::UnitProvenance>::new();
  let mut target = StaticRelation::<(usize, usize), unit::UnitProvenance>::new();

  // Initial
  source.insert_untagged(&mut ctx, vec![(0usize, 1usize), (1, 2)]);

  // Iterate until fixpoint
  while source.changed(&ctx) || target.changed(&ctx) {
    target.insert_dataflow_recent(&ctx, &source, true);
  }

  expect_static_collection(&target.complete(&ctx), vec![(0, 1), (1, 2)]);
}
