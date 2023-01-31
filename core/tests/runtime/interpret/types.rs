use scallop_core::common::tuple_type::TupleType;
use scallop_core::common::value_type::{FromType, ValueType};

#[test]
fn test_tuple_type_1() {
  let ty = <TupleType as FromType<(i8, i8, i8)>>::from_type();
  assert_eq!(
    ty,
    TupleType::Tuple(Box::new([
      TupleType::Value(ValueType::I8),
      TupleType::Value(ValueType::I8),
      TupleType::Value(ValueType::I8)
    ]))
  )
}
