use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;
use scallop_core::utils::*;

#[test]
fn test_min_max_with_topkproofs() {
  let prov = top_k_proofs::TopKProofsProvenance::<RcFamily>::new(3, false);
  expect_interpret_result_with_tag(
    r#"
      rel number = {0.1::1, 0.1::3, 0.8::5, 0.0::7}
      rel min_num = min(n: number(n))
    "#,
    prov,
    (
      "min_num",
      vec![(0.1f64, (1i32,)), (0.09, (3,)), (0.648, (5,)), (0.0, (7,))],
    ),
    top_k_proofs::TopKProofsProvenance::<RcFamily>::cmp,
  );
}

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

#[test]
fn test_unknown_keyword() {
  expect_front_compile_failure(
    r#"
      rel strings = {"hello", "world"}
      rel dummy() = x := string_join<unknown_keyword>(y: strings(y))
    "#,
    |s| s.contains("Unknown named parameter `unknown_keyword`"),
  )
}

#[test]
fn test_string_join_all() {
  let prov = min_max_prob::MinMaxProbProvenance::new();
  expect_interpret_result_with_tag(
    r#"
      rel symbols = {0.9::(0, "2"); 0.1::(0, "3")}
      rel symbols = {1.0::(1, "*")}
      rel symbols = {1.0::(2, "5")}
      rel formula(s) = s := string_join<all>[i](s: symbols(i, s))
    "#,
    prov,
    (
      "formula",
      vec![(0.9, ("2*5".to_string(),)), (0.1, ("3*5".to_string(),))],
    ),
    min_max_prob::MinMaxProbProvenance::cmp,
  );
}

#[test]
fn test_softmax() {
  let prov = min_max_prob::MinMaxProbProvenance::new();
  expect_interpret_result_with_tag(
    r#"
      rel symbols = {0.1::(0, "2"); 0.1::(0, "3")}
      rel softmax_symbols(i, s) = s := softmax(s: symbols(i, s))
    "#,
    prov,
    (
      "softmax_symbols",
      vec![(0.5, (0i32, "2".to_string())), (0.5, (0i32, "3".to_string()))],
    ),
    min_max_prob::MinMaxProbProvenance::cmp,
  );
}

#[test]
fn test_normalize_with_group_by() {
  let prov = min_max_prob::MinMaxProbProvenance::new();
  expect_interpret_result_with_tag(
    r#"
      rel symbols = {0.1::(0, "2"); 0.1::(0, "3")}
      rel symbols = {0.1::(1, "+"); 0.3::(1, "-")}
      rel normalized_symbols(i, s) = s := normalize(s: symbols(i, s))
    "#,
    prov,
    (
      "normalized_symbols",
      vec![
        (0.5, (0i32, "2".to_string())),
        (0.5, (0i32, "3".to_string())),
        (0.25, (1i32, "+".to_string())),
        (0.75, (1i32, "-".to_string())),
      ],
    ),
    min_max_prob::MinMaxProbProvenance::cmp,
  );
}

#[test]
fn test_rank_1() {
  let prov = min_max_prob::MinMaxProbProvenance::new();
  expect_interpret_result_with_tag(
    r#"
      rel digit = {0.1::0, 0.3::1, 0.5::2, 0.1::3}
      rel digit_rank(i, d) = (i, d) := rank(d: digit(d))
    "#,
    prov,
    (
      "digit_rank",
      vec![(0.5, (0usize, 2i32)), (0.3, (1, 1)), (0.1, (2, 0)), (0.1, (3, 3))],
    ),
    min_max_prob::MinMaxProbProvenance::cmp,
  );
}

#[test]
fn test_rank_descending_1() {
  let prov = min_max_prob::MinMaxProbProvenance::new();
  expect_interpret_result_with_tag(
    r#"
      rel digit = {0.1::0, 0.3::1, 0.5::2, 0.1::3}
      rel digit_rank(i, d) = (i, d) := rank<desc>(d: digit(d))
    "#,
    prov,
    (
      "digit_rank",
      vec![(0.5, (0usize, 2i32)), (0.3, (1, 1)), (0.1, (2, 0)), (0.1, (3, 3))],
    ),
    min_max_prob::MinMaxProbProvenance::cmp,
  );
}

#[test]
fn test_rank_ascending_1() {
  let prov = min_max_prob::MinMaxProbProvenance::new();
  expect_interpret_result_with_tag(
    r#"
      rel digit = {0.1::0, 0.3::1, 0.5::2, 0.1::3}
      rel digit_rank(i, d) = (i, d) := rank<asc>(d: digit(d))
    "#,
    prov,
    (
      "digit_rank",
      vec![(0.1, (0usize, 0i32)), (0.1, (1, 3)), (0.3, (2, 1)), (0.5, (3, 2))],
    ),
    min_max_prob::MinMaxProbProvenance::cmp,
  );
}

#[test]
fn test_rank_ascending_2() {
  let prov = min_max_prob::MinMaxProbProvenance::new();
  expect_interpret_result_with_tag(
    r#"
      rel digit = {0.1::0, 0.3::1, 0.5::2, 0.1::3}
      rel digit_rank(i, d) = (i, d) := rank<desc=false>(d: digit(d))
    "#,
    prov,
    (
      "digit_rank",
      vec![(0.1, (0usize, 0i32)), (0.1, (1, 3)), (0.3, (2, 1)), (0.5, (3, 2))],
    ),
    min_max_prob::MinMaxProbProvenance::cmp,
  );
}

#[test]
fn test_enumerate_1() {
  expect_interpret_result(
    r#"
      rel student = {"tom", "jerry"}
      rel student_with_id(id, name) = (id, name) := enumerate(name: student(name))
    "#,
    (
      "student_with_id",
      vec![(0usize, "jerry".to_string()), (1usize, "tom".to_string())],
    ),
  );
}

#[test]
fn test_reduce_rule_sugar_1() {
  expect_interpret_result(
    r#"
      rel student = {"tom", "jerry"}
      rel top_student = top<1>(n: student(n))
    "#,
    ("top_student", vec![("jerry".to_string(),)]),
  );
}

#[test]
fn test_argmax_sugar_1() {
  expect_interpret_result(
    r#"
      rel score = {("tom", 90), ("jerry", 80)}
      rel best_student = argmax[p](s: score(p, s))
    "#,
    ("best_student", vec![("tom".to_string(),)]),
  );
}

#[test]
fn test_reduce_rule_sugar_with_bad_aggregate_1() {
  expect_front_compile_failure(
    r#"
      rel score = {("tom", 90), ("jerry", 80)}
      rel best_student = argmax(s: score(p, s))
    "#,
    |s| s.contains("arity mismatch. Expected non-empty argument variables, but found 0"),
  );
}

#[test]
fn test_sort_1() {
  expect_interpret_result(
    r#"
type student_score(stu_id: String, name: String, score: f32)
rel student_score = {("1357", "jam", 10.0), ("2468", "tommy", 6.0)}
rel result(rank, score) = (rank, score) := sort<desc>(score: student_score(stu_id, _, score))
    "#,
    ("result", vec![(0usize, 10.0f32), (1usize, 6.0f32)]),
  );
}

#[test]
fn test_sort_2() {
  expect_interpret_result(
    r#"
type student_score(stu_id: String, name: String, score: f32)
rel student_score = {("1357", "jam", 10.0), ("2468", "tommy", 6.0)}
rel result(rank, stu_id) = (rank, stu_id, score) := sort<desc>[stu_id](score: student_score(stu_id, _, score))
    "#,
    (
      "result",
      vec![(0usize, "1357".to_string()), (1usize, "2468".to_string())],
    ),
  );
}

#[test]
fn test_sort_3() {
  expect_interpret_result(
    r#"
type student_score(stu_id: String, name: String, score: f32)
rel student_score = {("1357", "jam", 10.0), ("2468", "tommy", 6.0)}
rel result = argsort<desc>[stu_id](score: student_score(stu_id, _, score))
    "#,
    (
      "result",
      vec![(0usize, "1357".to_string()), (1usize, "2468".to_string())],
    ),
  );
}
