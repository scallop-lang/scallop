use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn test_avg_1() {
  expect_interpret_result(
    r#"
      rel scores = {(0, 55.0), (1, 45.0), (2, 50.0)}
      rel avg_score(a) = a := avg[i](s: scores(i, s))
    "#,
    ("avg_score", vec![(50.0f32,)]),
  );
}

#[test]
fn test_weighted_avg_1() {
  expect_interpret_result(
    r#"
      rel scores = {(0, 55.0), (1, 45.0), (2, 50.0)}
      rel avg_score(a) = a := weighted_avg[i](s: scores(i, s))
    "#,
    ("avg_score", vec![(50.0f32,)]),
  );
}

#[test]
fn test_weighted_avg_2() {
  let prov = min_max_prob::MinMaxProbProvenance::new();
  expect_interpret_result_with_tag(
    r#"
      rel scores = {0.5::(0, 55.0), 1.0::(1, 45.0), 0.5::(2, 50.0)}
      rel avg_score(a) = a := weighted_avg![i](s: scores(i, s))
    "#,
    prov,
    ("avg_score", vec![(1.0, (48.75f32,))]),
    min_max_prob::MinMaxProbProvenance::cmp,
  );
}

#[test]
fn test_string_join_1() {
  expect_interpret_result(
    r#"
      rel my_strings = {"hello", "world"}
      rel result(j) = j := string_join(s: my_strings(s))
    "#,
    ("result", vec![("helloworld".to_string(),)]),
  );
}

#[test]
fn test_string_join_2() {
  expect_interpret_result(
    r#"
      rel my_strings = {"hello", "world"}
      rel result(j) = j := string_join<" ">(s: my_strings(s))
    "#,
    ("result", vec![("hello world".to_string(),)]),
  );
}

#[test]
fn test_string_join_3() {
  expect_interpret_result(
    r#"
      rel my_strings = {(2, "hello"), (1, "world")}
      rel result(j) = j := string_join<" ">[i](s: my_strings(i, s))
    "#,
    ("result", vec![("world hello".to_string(),)]),
  );
}

#[test]
fn test_aggregate_compile_fail_1() {
  expect_front_compile_failure(
    r#"
      rel num = {1, 2, 3, 4}
      rel my_top(x) = x := top<1, 3, 5, 7>(y: num(y))
    "#,
    |s| s.contains("expected at most 1 parameter"),
  )
}

#[test]
fn test_aggregate_compile_fail_2() {
  expect_front_compile_failure(
    r#"
      rel num = {1, 2, 3, 4}
      rel my_top(x) = x := argmin(y: num(y))
    "#,
    |s| s.contains("Expected non-empty argument variables"),
  )
}
