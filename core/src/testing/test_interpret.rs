use crate::common::tuple::Tuple;
use crate::integrate::interpret_string;

use super::*;

pub fn expect_interpret_result<T: Into<Tuple> + Clone>(s: &str, (p, e): (&str, Vec<T>)) {
  let actual = interpret_string(s.to_string()).expect("Compile Error");
  expect_output_collection(&actual[p], e);
}

pub fn expect_interpret_empty_result(s: &str, p: &str) {
  let actual = interpret_string(s.to_string()).expect("Compile Error");
  assert!(actual[p].is_empty(), "The relation `{}` is not empty", p)
}

pub fn expect_interpret_multi_result(s: &str, expected: Vec<(&str, TestCollection)>) {
  let actual = interpret_string(s.to_string()).expect("Compile Error");
  for (p, a) in expected {
    expect_output_collection(&actual[p], a);
  }
}
