use scallop_core::common::{tuple_type::TupleType, value_type::FromType};

#[test]
fn test_tuple_type_1() {
  let ty = <TupleType as FromType<(i8, i8, i8)>>::from_type();
  println!("{:?}", ty);
}
