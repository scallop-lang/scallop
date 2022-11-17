use scallop_core::testing::*;

#[test]
fn test_prim_string_length_1() {
  expect_interpret_result(
    r#"
      rel strings = {"hello", "world!"}
      rel lengths(x, $string_length(x)) = strings(x)
    "#,
    (
      "lengths",
      vec![("hello".to_string(), 5usize), ("world!".to_string(), 6)],
    ),
  );
}

#[test]
fn test_prim_string_length_2() {
  expect_interpret_result(
    r#"
      rel strings = {"hello", "world!"}
      rel lengths(x, y) = strings(x), y == $string_length(x)
    "#,
    (
      "lengths",
      vec![("hello".to_string(), 5usize), ("world!".to_string(), 6)],
    ),
  );
}

#[test]
fn test_prim_string_concat_2() {
  expect_interpret_result(
    r#"
      rel strings = {"hello", "world!"}
      rel cat(x) = strings(a), strings(b), a != b, x == $string_concat(a, " ", b)
    "#,
    (
      "cat",
      vec![("hello world!".to_string(),), ("world! hello".to_string(),)],
    ),
  );
}

#[test]
fn test_prim_hash_1() {
  expect_interpret_result(
    r#"
      rel result(x) = x == $hash(1, 3)
    "#,
    ("result", vec![(5856262838373339618u64,)]),
  );
}

#[test]
fn test_prim_hash_2() {
  expect_interpret_result(
    r#"
      rel result($hash(1, 3))
    "#,
    ("result", vec![(5856262838373339618u64,)]),
  );
}

#[test]
fn test_prim_abs_1() {
  expect_interpret_result(
    r#"
      rel my_rel = {-1, 3, 5, -6}
      rel abs_result($abs(x)) = my_rel(x)
    "#,
    ("abs_result", vec![(1i32,), (3,), (5,), (6,)]),
  );
}

#[test]
fn test_prim_abs_2() {
  expect_interpret_result(
    r#"
      rel my_rel = {-1.5, 3.3, 5.0, -6.5}
      rel abs_result($abs(x)) = my_rel(x)
    "#,
    ("abs_result", vec![(1.5f32,), (3.3,), (5.0,), (6.5,)]),
  );
}

#[test]
fn test_prim_substring_1() {
  expect_interpret_result(
    r#"
      rel my_rel = {"hello world!"}
      rel result($substring(x, 0, 5)) = my_rel(x)
    "#,
    ("result", vec![("hello".to_string(),)]),
  );
}

#[test]
fn test_prim_substring_2() {
  expect_interpret_result(
    r#"
      rel my_rel = {"hello world!"}
      rel result($substring(x, 6)) = my_rel(x)
    "#,
    ("result", vec![("world!".to_string(),)]),
  );
}
