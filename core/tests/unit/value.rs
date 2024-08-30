use std::convert::*;

use scallop_core::common::value::*;

#[test]
fn value_try_into_1() {
  let v = Value::USize(10);
  let p: usize = v.try_into().unwrap_or(0);
  assert_eq!(p, 10);
}

#[test]
fn value_try_into_2() {
  let v = Value::I8(10);
  let p: usize = v.try_into().unwrap_or(0);
  assert_eq!(p, 0);
}
