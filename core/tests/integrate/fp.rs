use scallop_core::testing::*;

#[test]
fn range_free_1() {
  expect_interpret_result(
    r#"
      rel result(y) = range<usize>(0, 5, y)
    "#,
    ("result", vec![(0usize,), (1,), (2,), (3,), (4,)]),
  );
}

#[test]
fn range_constraint_1() {
  expect_interpret_result(
    r#"
      rel base = {("A", "B", 3.0)}
      rel result(a, b) = base(a, b, x) and soft_eq<f32>(x, 3.0)
    "#,
    ("result", vec![("A".to_string(), "B".to_string())]),
  );
}

#[test]
fn range_join_1() {
  expect_interpret_result(
    r#"
      rel base = {3}
      rel result(y) = base(x) and range<usize>(0, x, y)
    "#,
    ("result", vec![(0usize,), (1,), (2,)]),
  );
}

#[test]
fn range_join_2() {
  expect_interpret_result(
    r#"
      rel base = {3}
      rel result() = base(x) and range<usize>(0, x, 2)
    "#,
    ("result", vec![()]),
  );
}

#[test]
fn range_join_3() {
  expect_interpret_empty_result(
    r#"
      rel base = {3}
      rel result() = base(x) and range<usize>(0, x, 100)
    "#,
    "result",
  );
}

#[test]
fn range_join_4() {
  expect_interpret_result(
    r#"
      rel base = {3, 10}
      rel result(x) = base(x) and range<usize>(0, x, 5)
    "#,
    ("result", vec![(10usize,)]),
  );
}

#[test]
fn range_dt_1() {
  expect_interpret_empty_result(
    r#"
      type f(bound i: usize, bound j: usize)
      rel f(i, j) = range<usize>(i + 1, j, k) and c(i, k) and f(k + 1, j)
      type c(bound i: usize, bound j: usize)
    "#,
    "f",
  )
}

#[test]
fn string_chars_1() {
  expect_interpret_result(
    r#"
      rel string = {"hello"}
      rel result(i, c) = string(s), string_chars(s, i, c)
    "#,
    ("result", vec![(0usize, 'h'), (1, 'e'), (2, 'l'), (3, 'l'), (4, 'o')]),
  );
}

#[test]
fn floating_point_eq_1() {
  expect_interpret_multi_result(
    r#"
      rel result_1() = float_eq<f32>(3.000001, 1.000001 + 2.000001)
      rel result_2() = 3.000001 == 1.000001 + 2.000001
    "#,
    vec![("result_1", vec![()].into()), ("result_2", TestCollection::empty())],
  )
}

#[test]
fn string_split_1() {
  expect_interpret_multi_result(
    r#"
      rel string = {"abcde ab cde abcde"}
      rel pattern1 = {" "}
      rel pattern2 = {"ab"}
      rel pattern3 = {"abcde"}
      rel result1(o) = string(s), pattern1(p), string_split(s, p, o)
      rel result2(o) = string(s), pattern2(p), string_split(s, p, o)
      rel result3(o) = string(s), pattern3(p), string_split(s, p, o)
    "#,
    vec![
      (
        "result1",
        vec![
          ("abcde".to_string(),),
          ("ab".to_string(),),
          ("cde".to_string(),),
          ("abcde".to_string(),),
        ]
        .into(),
      ),
      (
        "result2",
        vec![
          ("".to_string(),),
          ("cde ".to_string(),),
          (" cde ".to_string(),),
          ("cde".to_string(),),
        ]
        .into(),
      ),
      (
        "result3",
        vec![("".to_string(),), (" ab cde ".to_string(),), ("".to_string(),)].into(),
      ),
    ],
  );
}

#[test]
fn string_find_1() {
  expect_interpret_multi_result(
    r#"
      rel string = {"abcde ab cde abcde"}
      rel pattern1 = {" "}
      rel pattern2 = {"ab"}
      rel pattern3 = {"cde"}
      rel result1(i, j) = string(s), pattern1(p), string_find(s, p, i, j)
      rel result2(i, j) = string(s), pattern2(p), string_find(s, p, i, j)
      rel result3(i, j) = string(s), pattern3(p), string_find(s, p, i, j)
    "#,
    vec![
      ("result1", vec![(5usize, 6usize), (8, 9), (12, 13)].into()),
      ("result2", vec![(0usize, 2usize), (6, 8), (13, 15)].into()),
      ("result3", vec![(2usize, 5usize), (9, 12), (15, 18)].into()),
    ],
  );
}

#[test]
fn datetime_ymd_1() {
  expect_interpret_result(
    r#"
      rel datetime = {t"2023-04-17T00:00:00Z"}
      rel result(y, m, d) = datetime(dt), datetime_ymd(dt, y, m, d)
    "#,
    ("result", vec![(2023i32, 4u32, 17u32)]),
  );
}

#[test]
fn range_syntax_sugar_1() {
  expect_interpret_multi_result(
    r#"
      rel result_1(x) = x in 0..5
      rel result_2(x) = x in 3..=6
    "#,
    vec![
      ("result_1", vec![(0i32,), (1,), (2,), (3,), (4,)].into()),
      ("result_2", vec![(3i32,), (4,), (5,), (6,)].into()),
    ],
  );
}

#[test]
fn range_syntax_sugar_2() {
  expect_interpret_result(
    r#"
      rel grid(x, y) = x in 2..5 and y in 5..=7
    "#,
    (
      "grid",
      vec![
        (2i32, 5i32),
        (2, 6),
        (2, 7),
        (3, 5),
        (3, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (4, 7),
      ],
    ),
  );
}
