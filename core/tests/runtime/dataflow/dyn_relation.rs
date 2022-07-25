use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn simple_relation_dataflow() {
  let mut ctx = unit::UnitContext;

  // Relations
  let mut source = DynamicRelation::<unit::Unit>::new();
  let mut target = DynamicRelation::<unit::Unit>::new();

  // Initial
  source.insert_untagged(&mut ctx, vec![(0usize, 1usize), (1usize, 2usize)]);

  // Iterate until fixpoint
  while source.changed(&ctx) || target.changed(&ctx) {
    target.insert_dataflow_recent(&ctx, &DynamicDataflow::dynamic_relation(&source));
  }

  expect_collection(
    &target.complete(&ctx),
    vec![(0usize, 1usize), (1usize, 2usize)],
  );
}
