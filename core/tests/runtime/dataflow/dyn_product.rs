use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn test_dynamic_product_1() {
  let mut ctx = unit::UnitProvenance;
  let mut rt = RuntimeEnvironment::new_std();

  // Relations
  let mut source_1 = DynamicRelation::<unit::UnitProvenance>::new();
  let mut source_2 = DynamicRelation::<unit::UnitProvenance>::new();
  let mut target = DynamicRelation::<unit::UnitProvenance>::new();

  // Initial
  source_1.insert_untagged(&mut ctx, vec![(0i8, 1i8), (1i8, 2i8)]);
  source_2.insert_untagged(&mut ctx, vec![(0i8, 2i8), (1i8, 5i8)]);

  // Iterate until fixpoint
  while source_1.changed(&ctx, rt.get_default_scheduler())
    || source_2.changed(&ctx, rt.get_default_scheduler())
    || target.changed(&ctx, rt.get_default_scheduler())
  {
    target.insert_dataflow_recent(
      &ctx,
      &DynamicDataflow::dynamic_relation(&source_1).product(DynamicDataflow::dynamic_relation(&source_2), &ctx),
      &mut rt,
    )
  }

  expect_collection(
    &target.complete(&ctx),
    vec![
      ((0i8, 1i8), (0i8, 2i8)),
      ((0i8, 1i8), (1i8, 5i8)),
      ((1i8, 2i8), (0i8, 2i8)),
      ((1i8, 2i8), (1i8, 5i8)),
    ],
  );
}
