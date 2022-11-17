use crate::common::tuple::Tuple;
use crate::integrate::*;
use crate::runtime::provenance::*;

use super::*;

pub fn expect_interpret_result<T: Into<Tuple> + Clone>(s: &str, (p, e): (&str, Vec<T>)) {
  let actual = interpret_string(s.to_string()).expect("Compile Error");
  expect_output_collection(&actual[p], e);
}

pub fn expect_interpret_result_with_tag<Prov, T, F>(s: &str, ctx: &mut Prov, (p, e): (&str, Vec<(Prov::OutputTag, T)>), f: F)
where
  Prov: Provenance,
  T: Into<Tuple> + Clone,
  F: Fn(&Prov::OutputTag, &Prov::OutputTag) -> bool,
{
  let actual = interpret_string_with_ctx(s.to_string(), ctx).expect("Interpret Error");
  expect_output_collection_with_tag(&actual[p], e, f);
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
