use scallop_core::common::value::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

static CARGO_MANIFEST_DIR: &'static str = env!("CARGO_MANIFEST_DIR");

#[test]
fn io_edge() {
  expect_interpret_result(
    &format!(
      r#"
        @file("{}/res/testing/csv/edge.csv")
        type edge(a: i32, b: i32)
        rel path(a, b) = edge(a, b) \/ path(a, c) /\ edge(c, b)
        query path
      "#,
      CARGO_MANIFEST_DIR,
    ),
    ("path", vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
  );
}

#[test]
fn io_edge_with_header() {
  expect_interpret_result(
    &format!(
      r#"
        @file("{}/res/testing/csv/edge_with_header.csv", header=true)
        type edge(a: i32, b: i32)
        rel path(a, b) = edge(a, b) \/ path(a, c) /\ edge(c, b)
        query path
      "#,
      CARGO_MANIFEST_DIR,
    ),
    ("path", vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
  );
}

#[test]
fn io_edge_with_deliminator() {
  expect_interpret_result(
    &format!(
      r#"
        @file("{}/res/testing/csv/edge_with_deliminator.csv", deliminator="\t")
        type edge(a: i32, b: i32)
        rel path(a, b) = edge(a, b) \/ path(a, c) /\ edge(c, b)
        query path
      "#,
      CARGO_MANIFEST_DIR,
    ),
    ("path", vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
  );
}

#[test]
fn io_edge_with_deliminator_and_header() {
  expect_interpret_result(
    &format!(
      r#"
        @file("{}/res/testing/csv/edge_with_deliminator_and_header.csv", deliminator="\t", header=true)
        type edge(a: i32, b: i32)
        rel path(a, b) = edge(a, b) \/ path(a, c) /\ edge(c, b)
        query path
      "#,
      CARGO_MANIFEST_DIR,
    ),
    ("path", vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
  );
}

#[test]
fn io_edge_with_prob() {
  let ctx = min_max_prob::MinMaxProbProvenance::new();
  expect_interpret_result_with_tag(
    &format!(
      r#"
        @file("{}/res/testing/csv/edge_with_prob.csv", has_probability=true)
        type edge(a: i32, b: i32)
      "#,
      CARGO_MANIFEST_DIR,
    ),
    ctx,
    ("edge", vec![(0.01, (0, 1)), (0.5, (1, 2)), (0.91, (2, 3))]),
    f64::eq,
  );
}

#[test]
fn io_student() {
  expect_interpret_result(
    &format!(
      r#"
        @file("{}/res/testing/csv/student.csv", keys="id")
        type student(id: String, field: Symbol, value: String)
      "#,
      CARGO_MANIFEST_DIR,
    ),
    (
      "student",
      vec![
        ("1".to_string(), Value::symbol_str("name"), "alice".to_string()),
        ("1".to_string(), Value::symbol_str("year"), "2022".to_string()),
        ("1".to_string(), Value::symbol_str("gender"), "female".to_string()),
        ("2".to_string(), Value::symbol_str("name"), "bob".to_string()),
        ("2".to_string(), Value::symbol_str("year"), "2023".to_string()),
        ("2".to_string(), Value::symbol_str("gender"), "male".to_string()),
      ],
    ),
  );
}

#[test]
fn io_student_with_fields() {
  expect_interpret_result(
    &format!(
      r#"
        @file("{}/res/testing/csv/student.csv", fields=["id", "name", "year"])
        type student(id: String, name: String, year: i32)
      "#,
      CARGO_MANIFEST_DIR,
    ),
    (
      "student",
      vec![
        ("1".to_string(), "alice".to_string(), 2022i32),
        ("2".to_string(), "bob".to_string(), 2023i32),
      ],
    ),
  );
}

#[test]
fn io_enrollment() {
  expect_interpret_result(
    &format!(
      r#"
        @file("{}/res/testing/csv/enrollment.csv", keys=["student_id", "course_id"])
        type enrollment(student_id: String, course_id: String, field: Symbol, value: String)
      "#,
      CARGO_MANIFEST_DIR,
    ),
    (
      "enrollment",
      vec![
        (
          "1".to_string(),
          "cse100".to_string(),
          Value::symbol_str("semester"),
          "fa".to_string(),
        ),
        (
          "1".to_string(),
          "cse100".to_string(),
          Value::symbol_str("year"),
          "2020".to_string(),
        ),
        (
          "1".to_string(),
          "cse100".to_string(),
          Value::symbol_str("grade"),
          "a".to_string(),
        ),
        (
          "1".to_string(),
          "cse102".to_string(),
          Value::symbol_str("semester"),
          "sp".to_string(),
        ),
        (
          "1".to_string(),
          "cse102".to_string(),
          Value::symbol_str("year"),
          "2021".to_string(),
        ),
        (
          "1".to_string(),
          "cse102".to_string(),
          Value::symbol_str("grade"),
          "a".to_string(),
        ),
        (
          "2".to_string(),
          "cse100".to_string(),
          Value::symbol_str("semester"),
          "sp".to_string(),
        ),
        (
          "2".to_string(),
          "cse100".to_string(),
          Value::symbol_str("year"),
          "2020".to_string(),
        ),
        (
          "2".to_string(),
          "cse100".to_string(),
          Value::symbol_str("grade"),
          "b".to_string(),
        ),
      ],
    ),
  );
}

#[test]
fn io_enrollment_with_keys_and_fields() {
  expect_interpret_result(
    &format!(
      r#"
        @file("{}/res/testing/csv/enrollment.csv", keys=["student_id", "course_id"], fields=["grade"])
        type enrollment(student_id: String, course_id: String, field: Symbol, value: String)
      "#,
      CARGO_MANIFEST_DIR,
    ),
    (
      "enrollment",
      vec![
        (
          "1".to_string(),
          "cse100".to_string(),
          Value::symbol_str("grade"),
          "a".to_string(),
        ),
        (
          "1".to_string(),
          "cse102".to_string(),
          Value::symbol_str("grade"),
          "a".to_string(),
        ),
        (
          "2".to_string(),
          "cse100".to_string(),
          Value::symbol_str("grade"),
          "b".to_string(),
        ),
      ],
    ),
  );
}

#[test]
fn io_enrollment_arity_error() {
  expect_interpret_specific_failure(
    &format!(
      r#"
        @file("{}/res/testing/csv/enrollment.csv", keys=["student_id", "course_id"], fields=["grade"])
        type enrollment(student_id: String, course_id: String)
      "#,
      CARGO_MANIFEST_DIR,
    ),
    |err| format!("{}", err).contains("IO: Arity mismatch"),
  );
}

#[test]
fn io_enrollment_type_error() {
  expect_interpret_specific_failure(
    &format!(
      r#"
        @file("{}/res/testing/csv/enrollment.csv", keys=["student_id", "course_id"], fields=["grade"])
        type enrollment(student_id: String, course_id: String, field: String, value: String)
      "#,
      CARGO_MANIFEST_DIR,
    ),
    |err| format!("{}", err).contains("IO: Expect `Symbol` type for field; found `String`"),
  );
}
