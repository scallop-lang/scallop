use scallop_core::common::{tuple::Tuple, value::Value};

#[test]
fn test_tuples_1() {
  let tuple: Tuple = ().into();
  println!("{:?}", tuple);
  assert_eq!(tuple, Tuple::from(()));
}

#[test]
fn test_tuples_2() {
  let tuple: Tuple = (3i8, "1234").into();
  println!("{:?}", tuple);
  assert_eq!(
    tuple,
    Tuple::Tuple(Box::new([Tuple::Value(Value::I8(3)), Tuple::Value(Value::Str("1234"))]))
  );
}
