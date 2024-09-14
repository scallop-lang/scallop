use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn simple_relation_dataflow() {
  let mut ctx = unit::UnitProvenance;
  let mut rt = RuntimeEnvironment::default();
  let sche = Scheduler::LFP;

  // Relations
  let mut source = DynamicRelation::<unit::UnitProvenance>::new();
  let mut target = DynamicRelation::<unit::UnitProvenance>::new();

  // Initial
  source.insert_untagged(&mut ctx, vec![(0usize, 1usize), (1usize, 2usize)]);

  // Iterate until fixpoint
  while source.changed(&ctx, &sche) || target.changed(&ctx, &sche) {
    target.insert_dataflow_recent(&ctx, &DynamicDataflow::dynamic_relation(&source), &mut rt);
  }

  expect_collection(&target.complete(&ctx), vec![(0usize, 1usize), (1usize, 2usize)]);
}
