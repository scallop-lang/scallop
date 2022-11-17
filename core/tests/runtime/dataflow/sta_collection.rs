use scallop_core::runtime::provenance::*;
use scallop_core::runtime::statics::*;
use scallop_core::testing::*;

#[test]
fn test_static_collection_dataflow_1() {
  let mut ctx = unit::UnitProvenance::default();
  let col = StaticCollection::<(i8, i8), unit::UnitProvenance>::from_vec(
    vec![
      StaticElement::new((0, 1), unit::Unit),
      StaticElement::new((1, 2), unit::Unit),
      StaticElement::new((2, 3), unit::Unit),
    ],
    &mut ctx,
  );
  let mut other = StaticRelation::<(i8, i8), unit::UnitProvenance>::new();
  let mut first_time = true;
  while other.changed(&ctx) || first_time {
    other.insert_dataflow_recent(&ctx, dataflow::collection(&col, first_time), true);
    first_time = false;
  }
  let result = other.complete(&ctx);
  expect_static_collection(&result, vec![(0, 1), (1, 2), (2, 3)]);
}
